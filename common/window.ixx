module;

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

export module common.window;

import std;
import vulkan;

export struct GlfwWindowConfig {
    int width = 1024;
    int height = 768;
    std::string title;
    bool resizable = false;
};

export class GlfwWindow {
public:
    explicit GlfwWindow(GlfwWindowConfig config) {
        if (config.width <= 0 || config.height <= 0) {
            throw std::runtime_error("GLFW window dimensions must be positive");
        }
        if (config.title.empty()) {
            throw std::runtime_error("GLFW window title must not be empty");
        }
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }
        initialized_ = true;

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, config.resizable ? GLFW_TRUE : GLFW_FALSE);
        window_ = glfwCreateWindow(config.width, config.height, config.title.c_str(), nullptr, nullptr);
        if (window_ == nullptr) {
            glfwTerminate();
            initialized_ = false;
            throw std::runtime_error("Failed to create GLFW window");
        }
    }

    ~GlfwWindow() {
        if (window_ != nullptr) {
            glfwDestroyWindow(window_);
        }
        if (initialized_) {
            glfwTerminate();
        }
    }

    GlfwWindow(const GlfwWindow &) = delete;
    GlfwWindow &operator=(const GlfwWindow &) = delete;
    GlfwWindow(GlfwWindow &&) = delete;
    GlfwWindow &operator=(GlfwWindow &&) = delete;

    [[nodiscard]] std::vector<std::string> requiredInstanceExtensions() const {
        std::uint32_t extensionCount = 0;
        const char **extensions = glfwGetRequiredInstanceExtensions(&extensionCount);
        if (extensions == nullptr || extensionCount == 0) {
            throw std::runtime_error("GLFW did not provide Vulkan instance extensions");
        }

        std::vector<std::string> result;
        result.reserve(extensionCount);
        for (std::uint32_t index = 0; index < extensionCount; ++index) {
            result.emplace_back(extensions[index]);
        }
        return result;
    }

    [[nodiscard]] vk::raii::SurfaceKHR createSurface(const vk::raii::Instance &instance) const {
        VkSurfaceKHR rawSurface = VK_NULL_HANDLE;
        const VkResult result =
            glfwCreateWindowSurface(static_cast<VkInstance>(*instance), window_, nullptr, &rawSurface);
        if (result != VK_SUCCESS) {
            throw std::runtime_error(std::format("Failed to create GLFW Vulkan surface: {}", static_cast<int>(result)));
        }
        return vk::raii::SurfaceKHR(instance, rawSurface);
    }

    [[nodiscard]] vk::Extent2D framebufferExtent() const {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window_, &width, &height);
        if (width < 0 || height < 0) {
            throw std::runtime_error("GLFW returned an invalid framebuffer extent");
        }
        return {
            .width = static_cast<std::uint32_t>(width),
            .height = static_cast<std::uint32_t>(height),
        };
    }

    [[nodiscard]] bool shouldClose() const {
        return glfwWindowShouldClose(window_) == GLFW_TRUE;
    }

    void pollEvents() const {
        glfwPollEvents();
    }

private:
    GLFWwindow *window_ = nullptr;
    bool initialized_ = false;
};
