#include <GLFW/glfw3.h>

#ifdef _WIN32
#include <windows.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#endif

#include <GLFW/glfw3native.h>

import vulkan_hpp;
import std;
import common;

class HelloTriangleApplication {
public:
    void run() {
        initWindow(1024, 768);
        initVulkan();
        mainLoop();
        cleanup();
    }
private:
    vk::raii::Context context;
    vk::raii::Instance instance = nullptr;
    vk::raii::Device device = nullptr;
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::detail::DispatchLoaderDynamic dldy;

    std::uint32_t graphicsQueueFamilyIndex = 0;

    vk::raii::CommandPool commandPool = nullptr;
    vk::raii::CommandBuffers commandBuffers = nullptr;  // must be destroyed before commandPool

    GLFWwindow* window = nullptr;

    vk::raii::SurfaceKHR surface = nullptr;
    vk::raii::SwapchainKHR swapchain = nullptr;
    vk::SurfaceFormatKHR swapchainSurfaceFormat{};
    vk::Extent2D swapchainExtent{};
    std::vector<vk::Image> swapchainImages;
    std::vector<vk::raii::ImageView> imageViews;
    std::vector<vk::raii::Image> depthImages;
    std::vector<vk::raii::DeviceMemory> depthImageMemorys;
    std::vector<vk::raii::ImageView> depthImageViews;

    vk::raii::RenderPass renderPass = nullptr;

    std::vector<vk::raii::Framebuffer> framebuffers;

    void initWindow(int width, int height) {
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(width, height, "Hello Triangle", nullptr, nullptr);
    }

    void initVulkan() {
        createContext();
        createInstance();
        createSurface();
        selectPhysicalDevice();
        createDevice();
        createSwapchain();
        createCommandPool();
        createCommandBuffer();
        createRenderPass();
        createFramebuffers();
    }
    void createContext() {
        auto instanceLayerProperties = context.enumerateInstanceLayerProperties();
        std::println("Instance supported layers: {}", instanceLayerProperties.size());
        for (auto &e : instanceLayerProperties) {
            std::println("{}", e.layerName.data());
        }
        auto instanceExtensionProperties = context.enumerateInstanceExtensionProperties();
        std::println("Instance supported extensions: {}", instanceExtensionProperties.size());
        for (auto &e : instanceExtensionProperties) {
            std::println("{}", e.extensionName.data());
        }
    }

    void createInstance() {
        // initialize minimal set of function pointers
        dldy.init();

        auto apiVersion = context.enumerateInstanceVersion();

        std::println("API Version: {}.{}.{}.{}", vk::apiVersionMajor(apiVersion), vk::apiVersionMinor(apiVersion),
                     vk::apiVersionPatch(apiVersion), vk::apiVersionVariant(apiVersion));

        vk::ApplicationInfo appInfo{.pApplicationName = "Hello Triangle",
                                    .applicationVersion = vk::makeApiVersion(1, 0, 0, 0),
                                    .pEngineName = "No Engine",
                                    .engineVersion = vk::makeApiVersion(1, 0, 0, 0),
                                    .apiVersion = apiVersion};
        //   .apiVersion = vk::ApiVersion14};

        std::vector<const char *> enabledLayers = {
            "VK_LAYER_KHRONOS_validation",
            "VK_LAYER_LUNARG_crash_diagnostic",
            // "VK_LAYER_LUNARG_api_dump",
        };

        std::vector<const char *> enabledExtensions = {
            "VK_KHR_surface",
            "VK_KHR_win32_surface",
            "VK_EXT_debug_utils",
            "VK_KHR_get_physical_device_properties2",
            "VK_KHR_get_surface_capabilities2",
            "VK_EXT_surface_maintenance1",
        };

        vk::InstanceCreateInfo createInfo{.pApplicationInfo = &appInfo,
                                          .enabledLayerCount = static_cast<std::uint32_t>(enabledLayers.size()),
                                          .ppEnabledLayerNames = enabledLayers.data(),
                                          .enabledExtensionCount = static_cast<std::uint32_t>(enabledExtensions.size()),
                                          .ppEnabledExtensionNames = enabledExtensions.data()};

        instance = vk::raii::Instance(context, createInfo);

        // load function pointers with created instance
        dldy.init(*instance);
    }
    void selectPhysicalDevice() {
        // create a dispatcher, based on additional vkDevice/vkGetDeviceProcAddr
        auto physicalDevices = instance.enumeratePhysicalDevices();

        debug_throw_if(physicalDevices.empty(), "Failed to find GPUs with Vulkan support!");

        for (auto &e : physicalDevices) {
            auto properties = e.getProperties();
            std::println("Physical Device: {}, version:{}.{}.{}, ", properties.deviceName.data(),
                         vk::apiVersionMajor(properties.apiVersion), vk::apiVersionMinor(properties.apiVersion),
                         vk::apiVersionPatch(properties.apiVersion));
        }
        for (auto &pd : physicalDevices) {
            auto properties = pd.getProperties();
            auto queueFamilyProperties = pd.getQueueFamilyProperties();
            for (std::uint32_t i = 0; i < queueFamilyProperties.size(); ++i) {
                if (!(queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics)) {
                    continue;
                }
                if (!pd.getSurfaceSupportKHR(i, *surface)) {
                    continue;
                }
                physicalDevice = pd;
                graphicsQueueFamilyIndex = i;
                std::println("Selected GPU: {}", properties.deviceName.data());
                return;
            }
        }
        debug_throw_if(true, "No suitable GPU found (graphics + surface present)!");
    }

    void createDevice() {
        auto deviceExtensionProperties = physicalDevice.enumerateDeviceExtensionProperties();
        std::println("Device supported extensions: {}", deviceExtensionProperties.size());
        for (auto &e : deviceExtensionProperties) {
            std::println("{}", e.extensionName.data());
        }

        std::vector<float> queuePriorities = {1.0f};
        vk::DeviceQueueCreateInfo queueCreateInfo{
            .queueFamilyIndex = static_cast<std::uint32_t>(graphicsQueueFamilyIndex),
            .queueCount = 1,
            .pQueuePriorities = queuePriorities.data(),
        };

        std::vector<const char*> enabledDeviceExtensions = {
            "VK_KHR_swapchain"
        };

        vk::DeviceCreateInfo deviceCreateInfo{
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queueCreateInfo,
            .enabledExtensionCount = static_cast<std::uint32_t>(enabledDeviceExtensions.size()),
            .ppEnabledExtensionNames = enabledDeviceExtensions.data(),
        };

        device = physicalDevice.createDevice(deviceCreateInfo);

        // OPTIONAL: load function pointers for a specific vk::Device
        dldy.init(*device);
    }

    void createCommandPool() {
        vk::CommandPoolCreateInfo commandPoolCreateInfo{
            .queueFamilyIndex = static_cast<std::uint32_t>(graphicsQueueFamilyIndex),
        };
        commandPool = device.createCommandPool(commandPoolCreateInfo);
    }

    void createCommandBuffer() {
        vk::CommandBufferAllocateInfo commandBufferAllocateInfo{
            .commandPool = *commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        };
        // commandBuffers is a vector of vk::raii::CommandBuffer
        commandBuffers = vk::raii::CommandBuffers(device, commandBufferAllocateInfo);
    }
    void createSurface() {
        vk::Win32SurfaceCreateInfoKHR win32SurfaceCreateInfo{
            .hinstance = GetModuleHandle(nullptr),
            .hwnd = glfwGetWin32Window(window),
        };

        surface = vk::raii::SurfaceKHR(instance, win32SurfaceCreateInfo);
    }

    static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats) {
        for (const auto &availableFormat : availableFormats) {
            if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
                availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return availableFormat;
            }
        }
        return availableFormats[0];
    }

    static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes) {
        for (const auto &availablePresentMode : availablePresentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                return availablePresentMode;
            }
        }
        return vk::PresentModeKHR::eFifo;
    }

    static vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities, int framebufferWidth,
                                         int framebufferHeight) {
        if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        vk::Extent2D actualExtent{static_cast<std::uint32_t>(framebufferWidth),
                                  static_cast<std::uint32_t>(framebufferHeight)};
        actualExtent.width =
            std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height =
            std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
        return actualExtent;
    }

    static std::uint32_t findMemoryTypeIndex(const vk::raii::PhysicalDevice &physicalDevice,
                                             const vk::MemoryRequirements &memoryRequirements,
                                             const vk::MemoryPropertyFlags &properties) {
        auto memoryProperties = physicalDevice.getMemoryProperties();
        for (std::uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
            if ((memoryRequirements.memoryTypeBits & (1 << i)) &&
                (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("Failed to find suitable memory type!");
    }

    void createSwapchain() {
        auto capabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
        auto formats = physicalDevice.getSurfaceFormatsKHR(*surface);
        auto presentModes = physicalDevice.getSurfacePresentModesKHR(*surface);

        debug_throw_if(formats.empty(), "no surface formats available!");
        debug_throw_if(presentModes.empty(), "no present modes available!");

        swapchainSurfaceFormat = chooseSwapSurfaceFormat(formats);
        vk::PresentModeKHR presentMode = chooseSwapPresentMode(presentModes);

        int fbWidth = 0;
        int fbHeight = 0;
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        swapchainExtent = chooseSwapExtent(capabilities, fbWidth, fbHeight);

        std::uint32_t imageCount = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
            imageCount = capabilities.maxImageCount;
        }

        std::println("Swapchain image count: {}", imageCount);

        vk::CompositeAlphaFlagBitsKHR compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
        if (!(capabilities.supportedCompositeAlpha & compositeAlpha)) {
            if (capabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::ePreMultiplied) {
                compositeAlpha = vk::CompositeAlphaFlagBitsKHR::ePreMultiplied;
            } else if (capabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::ePostMultiplied) {
                compositeAlpha = vk::CompositeAlphaFlagBitsKHR::ePostMultiplied;
            } else {
                compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eInherit;
            }
        }

        vk::SwapchainCreateInfoKHR createInfo{
            .surface = *surface,
            .minImageCount = imageCount,
            .imageFormat = swapchainSurfaceFormat.format,
            .imageColorSpace = swapchainSurfaceFormat.colorSpace,
            .imageExtent = swapchainExtent,
            .imageArrayLayers = 1,
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
            .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform = capabilities.currentTransform,
            .compositeAlpha = compositeAlpha,
            .presentMode = presentMode,
            .clipped = true,
        };

        swapchain = device.createSwapchainKHR(createInfo);
        swapchainImages = swapchain.getImages();
        std::println("Swapchain: {} images, extent {}x{}, format {}, present mode {}", swapchainImages.size(),
                     swapchainExtent.width, swapchainExtent.height, vk::to_string(swapchainSurfaceFormat.format),
                     vk::to_string(presentMode));


        //create image views
        for (auto &image : swapchainImages) {
            vk::ImageViewCreateInfo imageViewCreateInfo{
                .image = image,
                .viewType = vk::ImageViewType::e2D,
                .format = swapchainSurfaceFormat.format,
                .subresourceRange =
                    {
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
            };
            imageViews.push_back(vk::raii::ImageView(device, imageViewCreateInfo));

            // create depth image
            vk::ImageCreateInfo depthImageCreateInfo{
                .imageType = vk::ImageType::e2D,
                .format = vk::Format::eD32Sfloat,
                .extent = {swapchainExtent.width, swapchainExtent.height, 1},
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = vk::SampleCountFlagBits::e1,
                .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
            };
            depthImages.push_back(vk::raii::Image(device, depthImageCreateInfo));
            vk::MemoryRequirements memoryRequirements = depthImages.back().getMemoryRequirements();

            auto memoryTypeIndex =
                findMemoryTypeIndex(physicalDevice, memoryRequirements, vk::MemoryPropertyFlagBits::eDeviceLocal);

            vk::MemoryAllocateInfo memoryAllocateInfo{
                .allocationSize = memoryRequirements.size,
                .memoryTypeIndex = memoryTypeIndex,
            };
            depthImageMemorys.push_back(vk::raii::DeviceMemory(device, memoryAllocateInfo));
            depthImages.back().bindMemory(*depthImageMemorys.back(), 0);

            vk::ImageViewCreateInfo depthImageViewCreateInfo{
                .image = depthImages.back(),
                .viewType = vk::ImageViewType::e2D,
                .format = vk::Format::eD32Sfloat,
                .subresourceRange =
                    {
                        .aspectMask = vk::ImageAspectFlagBits::eDepth,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
            };
            depthImageViews.push_back(vk::raii::ImageView(device, depthImageViewCreateInfo));
        }
    }
    void createRenderPass() {
        std::array<vk::AttachmentDescription, 2> attachments = {
            {
                {
                    .format = swapchainSurfaceFormat.format,
                    .samples = vk::SampleCountFlagBits::e1,
                    .loadOp = vk::AttachmentLoadOp::eClear,
                    .storeOp = vk::AttachmentStoreOp::eStore,
                    .initialLayout = vk::ImageLayout::eUndefined,
                    .finalLayout = vk::ImageLayout::ePresentSrcKHR,
                },
                {
                    .format = vk::Format::eD32Sfloat,
                    .samples = vk::SampleCountFlagBits::e1,
                    .loadOp = vk::AttachmentLoadOp::eClear,
                    .storeOp = vk::AttachmentStoreOp::eStore,
                    .initialLayout = vk::ImageLayout::eUndefined,
                    .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                },
            },
        };
        std::array<vk::AttachmentReference, 1> colorAttachments = {vk::AttachmentReference{
            .attachment = 0,
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
        }};
        vk::AttachmentReference depthAttachment{
            .attachment = 0,
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
        };

        vk::SubpassDescription subpass{
            .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
            .inputAttachmentCount = 0,
            .pInputAttachments = nullptr,
            .colorAttachmentCount = static_cast<std::uint32_t>(colorAttachments.size()),
            .pColorAttachments = colorAttachments.data(),
        };

        vk::RenderPassCreateInfo renderPassCreateInfo{
            .attachmentCount = static_cast<std::uint32_t>(attachments.size()),
            .pAttachments = attachments.data(),
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 0,
            .pDependencies = nullptr,
        };
        renderPass = vk::raii::RenderPass(device, renderPassCreateInfo);
    }

    void createFramebuffers() {
        for (size_t i = 0; i < swapchainImages.size(); i++) {
            std::array<vk::ImageView, 2> attachments = {*imageViews[i], *depthImageViews[i]};
            vk::FramebufferCreateInfo framebufferCreateInfo{
                .renderPass = *renderPass,
                .attachmentCount = static_cast<std::uint32_t>(attachments.size()),
                .pAttachments = attachments.data(),
                .width = swapchainExtent.width,
                .height = swapchainExtent.height,
                .layers = 1,
            };
            framebuffers.push_back(vk::raii::Framebuffer(device, framebufferCreateInfo));
        }
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            glfwSwapBuffers(window);
        }
    }
    void cleanup() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
};

int main() {
    try {
        HelloTriangleApplication app;
        app.run();
    } catch (const std::exception &e) {
        std::println("Error: {}", e.what());
        return 1;
    }

    return 0;
}