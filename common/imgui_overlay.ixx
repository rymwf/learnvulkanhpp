module;

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

export module common.imgui_overlay;

import std;
import vulkan;
import common.window;

export struct ImGuiVulkanOverlayConfig {
    GlfwWindow *window = nullptr;
    std::uint32_t apiVersion = vk::ApiVersion10;
    vk::Instance instance{};
    vk::PhysicalDevice physicalDevice{};
    vk::Device device{};
    std::uint32_t queueFamilyIndex = 0;
    vk::Queue queue{};
    vk::RenderPass renderPass{};
    std::uint32_t minImageCount = 2;
    std::uint32_t imageCount = 2;
};

export class ImGuiVulkanOverlay {
public:
    ImGuiVulkanOverlay() = default;

    ~ImGuiVulkanOverlay() {
        shutdown();
    }

    ImGuiVulkanOverlay(const ImGuiVulkanOverlay &) = delete;
    ImGuiVulkanOverlay &operator=(const ImGuiVulkanOverlay &) = delete;
    ImGuiVulkanOverlay(ImGuiVulkanOverlay &&) = delete;
    ImGuiVulkanOverlay &operator=(ImGuiVulkanOverlay &&) = delete;

    void initialize(ImGuiVulkanOverlayConfig config) {
        if (context_ != nullptr) {
            throw std::logic_error("The ImGui overlay is already initialized");
        }
        validateConfig(config);
        config_ = config;

        try {
            IMGUI_CHECKVERSION();
            context_ = ImGui::CreateContext();
            if (context_ == nullptr) {
                throw std::runtime_error("Failed to create the ImGui context");
            }
            makeCurrent();

            auto &io = ImGui::GetIO();
            io.IniFilename = nullptr;
            io.LogFilename = nullptr;
            ImGui::StyleColorsDark();
            auto &style = ImGui::GetStyle();
            style.WindowRounding = 4.0f;
            style.FrameRounding = 3.0f;
            style.GrabRounding = 2.0f;

            if (!ImGui_ImplGlfw_InitForVulkan(config_.window->nativeHandle(), true)) {
                throw std::runtime_error("Failed to initialize the ImGui GLFW backend");
            }
            platformInitialized_ = true;
            initializeRenderer(config_.renderPass, config_.minImageCount, config_.imageCount);
        } catch (...) {
            shutdown();
            throw;
        }
    }

    void beginFrame() {
        requireRenderer();
        if (frameActive_) {
            throw std::logic_error("An ImGui frame is already active");
        }

        makeCurrent();
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        frameActive_ = true;
    }

    void render(vk::raii::CommandBuffer &commandBuffer) {
        requireRenderer();
        if (!frameActive_) {
            throw std::logic_error("No ImGui frame is active");
        }

        makeCurrent();
        ImGui::Render();
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), static_cast<VkCommandBuffer>(*commandBuffer));
        frameActive_ = false;
    }

    [[nodiscard]] bool wantsMouseCapture() const noexcept {
        if (context_ == nullptr) {
            return false;
        }
        makeCurrent();
        return ImGui::GetIO().WantCaptureMouse;
    }

    [[nodiscard]] bool wantsKeyboardCapture() const noexcept {
        if (context_ == nullptr) {
            return false;
        }
        makeCurrent();
        return ImGui::GetIO().WantCaptureKeyboard;
    }

    void releaseRenderer() noexcept {
        if (!rendererInitialized_) {
            return;
        }

        makeCurrent();
        endActiveFrame();
        ImGui_ImplVulkan_Shutdown();
        rendererInitialized_ = false;
    }

    void recreateRenderer(vk::RenderPass renderPass,
                          std::uint32_t minImageCount,
                          std::uint32_t imageCount) {
        if (context_ == nullptr || !platformInitialized_) {
            throw std::logic_error("The ImGui overlay is not initialized");
        }
        validateRendererConfig(renderPass, minImageCount, imageCount);

        releaseRenderer();
        config_.renderPass = renderPass;
        config_.minImageCount = minImageCount;
        config_.imageCount = imageCount;
        initializeRenderer(renderPass, minImageCount, imageCount);
    }

    void shutdown() noexcept {
        if (context_ == nullptr) {
            return;
        }

        makeCurrent();
        endActiveFrame();
        releaseRenderer();
        if (platformInitialized_) {
            ImGui_ImplGlfw_Shutdown();
            platformInitialized_ = false;
        }
        ImGui::DestroyContext(context_);
        context_ = nullptr;
        config_ = {};
    }

private:
    static void validateRendererConfig(vk::RenderPass renderPass,
                                       std::uint32_t minImageCount,
                                       std::uint32_t imageCount) {
        if (!renderPass) {
            throw std::invalid_argument("The ImGui render pass must be valid");
        }
        if (minImageCount < 2) {
            throw std::invalid_argument("The ImGui minimum image count must be at least two");
        }
        if (imageCount < minImageCount) {
            throw std::invalid_argument("The ImGui image count must not be smaller than its minimum");
        }
    }

    static void validateConfig(const ImGuiVulkanOverlayConfig &config) {
        if (config.window == nullptr || !config.window->nativeHandle()) {
            throw std::invalid_argument("The ImGui GLFW window must be valid");
        }
        if (!config.instance || !config.physicalDevice || !config.device || !config.queue) {
            throw std::invalid_argument("The ImGui Vulkan handles must be valid");
        }
        validateRendererConfig(config.renderPass, config.minImageCount, config.imageCount);
    }

    void initializeRenderer(vk::RenderPass renderPass,
                            std::uint32_t minImageCount,
                            std::uint32_t imageCount) {
        makeCurrent();
        ImGui_ImplVulkan_InitInfo initInfo{};
        initInfo.ApiVersion = config_.apiVersion;
        initInfo.Instance = static_cast<VkInstance>(config_.instance);
        initInfo.PhysicalDevice = static_cast<VkPhysicalDevice>(config_.physicalDevice);
        initInfo.Device = static_cast<VkDevice>(config_.device);
        initInfo.QueueFamily = config_.queueFamilyIndex;
        initInfo.Queue = static_cast<VkQueue>(config_.queue);
        initInfo.DescriptorPoolSize = 64;
        initInfo.MinImageCount = minImageCount;
        initInfo.ImageCount = imageCount;
        initInfo.PipelineInfoMain.RenderPass = static_cast<VkRenderPass>(renderPass);
        initInfo.PipelineInfoMain.Subpass = 0;
        initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

        if (!ImGui_ImplVulkan_Init(&initInfo)) {
            throw std::runtime_error("Failed to initialize the ImGui Vulkan backend");
        }
        rendererInitialized_ = true;
    }

    void requireRenderer() const {
        if (context_ == nullptr || !platformInitialized_ || !rendererInitialized_) {
            throw std::logic_error("The ImGui overlay renderer is not initialized");
        }
    }

    void makeCurrent() const noexcept {
        ImGui::SetCurrentContext(context_);
    }

    void endActiveFrame() noexcept {
        if (frameActive_) {
            ImGui::EndFrame();
            frameActive_ = false;
        }
    }

    ImGuiVulkanOverlayConfig config_{};
    ImGuiContext *context_ = nullptr;
    bool platformInitialized_ = false;
    bool rendererInitialized_ = false;
    bool frameActive_ = false;
};
