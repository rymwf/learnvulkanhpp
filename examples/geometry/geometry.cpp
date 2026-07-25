import vulkan;
import std;
import common;

#ifndef SHADER_SPV_DIR
#error "SHADER_SPV_DIR must be set by CMake (see examples/geometry/CMakeLists.txt)"
#endif

constexpr std::uint32_t maxConcurrentFrames = 2;

class GeometryShaderApplication {
public:
    GeometryShaderApplication()
        : window({
              .width = 1024,
              .height = 768,
              .title = "Vulkan Geometry Shader",
          }),
          vulkan({
              .applicationName = "Vulkan Geometry Shader",
              .apiVersion = vk::ApiVersion10,
              .requiredExtensions = window.requiredInstanceExtensions(),
          }),
          surface(window.createSurface(vulkan.instance())) {}

    void run() {
        initVulkan();
        mainLoop();
    }

private:
    GlfwWindow window;
    VulkanInstance vulkan;
    vk::raii::SurfaceKHR surface = nullptr;
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;

    std::uint32_t graphicsQueueFamilyIndex = 0;

    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;
    vk::raii::Queue graphicsQueue = nullptr;

    vk::raii::SwapchainKHR swapchain = nullptr;
    vk::SurfaceFormatKHR swapchainSurfaceFormat{};
    vk::Extent2D swapchainExtent{};
    std::vector<vk::Image> swapchainImages;
    std::vector<vk::raii::ImageView> imageViews;
    std::vector<vk::raii::Image> depthImages;
    std::vector<vk::raii::DeviceMemory> depthImageMemorys;
    std::vector<vk::raii::ImageView> depthImageViews;

    vk::raii::RenderPass renderPass = nullptr;
    vk::raii::ShaderModule vertShaderModule = nullptr;
    vk::raii::ShaderModule geometryShaderModule = nullptr;
    vk::raii::ShaderModule fragShaderModule = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;

    std::vector<vk::raii::Framebuffer> framebuffers;

    std::vector<vk::raii::Semaphore> imageAvailableSemaphores;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;
    std::uint32_t currentFrame = 0;

    void initVulkan() {
        selectPhysicalDevice();
        createDevice();
        createSwapchain();
        createCommandPool();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandBuffers();
        createSyncObjects();
    }
    void selectPhysicalDevice() {
        // create a dispatcher, based on additional vkDevice/vkGetDeviceProcAddr
        auto physicalDevices = vulkan.instance().enumeratePhysicalDevices();

        debug_throw_if(physicalDevices.empty(), "Failed to find GPUs with Vulkan support!");

        for (auto &e : physicalDevices) {
            auto properties = e.getProperties();
            std::println("Physical Device: {}, version:{}.{}.{}, ", properties.deviceName.data(),
                         vk::apiVersionMajor(properties.apiVersion), vk::apiVersionMinor(properties.apiVersion),
                         vk::apiVersionPatch(properties.apiVersion));
        }
        for (auto &pd : physicalDevices) {
            auto properties = pd.getProperties();
            if (!pd.getFeatures().geometryShader) {
                continue;
            }
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
        throw std::runtime_error("No suitable GPU found (geometry shader + graphics + surface present)");
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

        std::vector<const char *> enabledDeviceExtensions = {"VK_KHR_swapchain"};
        vk::PhysicalDeviceFeatures enabledFeatures{
            .geometryShader = true,
        };

        vk::DeviceCreateInfo deviceCreateInfo{
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queueCreateInfo,
            .enabledExtensionCount = static_cast<std::uint32_t>(enabledDeviceExtensions.size()),
            .ppEnabledExtensionNames = enabledDeviceExtensions.data(),
            .pEnabledFeatures = &enabledFeatures,
        };

        device = physicalDevice.createDevice(deviceCreateInfo);

        graphicsQueue = device.getQueue(graphicsQueueFamilyIndex, 0);
    }

    void createCommandPool() {
        vk::CommandPoolCreateInfo commandPoolCreateInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = static_cast<std::uint32_t>(graphicsQueueFamilyIndex),
        };
        commandPool = device.createCommandPool(commandPoolCreateInfo);
    }

    void createCommandBuffers() {
        vk::CommandBufferAllocateInfo commandBufferAllocateInfo{
            .commandPool = *commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = maxConcurrentFrames,
        };
        commandBuffers = device.allocateCommandBuffers(commandBufferAllocateInfo);
    }

    void recordCommandBuffer(std::uint32_t frameIndex, std::uint32_t imageIndex) {
        commandBuffers[frameIndex].reset({});

        vk::CommandBufferBeginInfo beginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
        commandBuffers[frameIndex].begin(beginInfo);

        std::array<vk::ClearValue, 2> clearValues{};
        clearValues[0].color.setFloat32(std::array<float, 4>{0.02f, 0.02f, 0.08f, 1.0f});
        clearValues[1].depthStencil = vk::ClearDepthStencilValue{1.0f, 0};

        vk::RenderPassBeginInfo renderPassInfo{
            .renderPass = *renderPass,
            .framebuffer = *framebuffers[imageIndex],
            .renderArea = {.offset = {0, 0}, .extent = swapchainExtent},
            .clearValueCount = static_cast<std::uint32_t>(clearValues.size()),
            .pClearValues = clearValues.data(),
        };

        commandBuffers[frameIndex].beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
        commandBuffers[frameIndex].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
        commandBuffers[frameIndex].draw(3, 1, 0, 0);
        commandBuffers[frameIndex].endRenderPass();
        commandBuffers[frameIndex].end();
    }

    void createSyncObjects() {
        imageAvailableSemaphores.clear();
        renderFinishedSemaphores.clear();
        inFlightFences.clear();
        imageAvailableSemaphores.reserve(maxConcurrentFrames);
        inFlightFences.reserve(maxConcurrentFrames);
        renderFinishedSemaphores.reserve(swapchainImages.size());

        for (std::uint32_t i = 0; i < maxConcurrentFrames; ++i) {
            imageAvailableSemaphores.push_back(device.createSemaphore({}));
            inFlightFences.push_back(device.createFence({.flags = vk::FenceCreateFlagBits::eSignaled}));
        }
        for (std::size_t i = 0; i < swapchainImages.size(); ++i) {
            renderFinishedSemaphores.push_back(device.createSemaphore({}));
        }
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

        const auto framebufferExtent = window.framebufferExtent();
        swapchainExtent = chooseSwapExtent(capabilities, static_cast<int>(framebufferExtent.width),
                                           static_cast<int>(framebufferExtent.height));

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

        // create image views
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
            imageViews.push_back(device.createImageView(imageViewCreateInfo));

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
            depthImages.push_back(device.createImage(depthImageCreateInfo));
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
            depthImageViews.push_back(device.createImageView(depthImageViewCreateInfo));
        }
    }
    void createRenderPass() {
        std::array<vk::AttachmentDescription, 2> attachments = {
            vk::AttachmentDescription{
                .format = swapchainSurfaceFormat.format,
                .samples = vk::SampleCountFlagBits::e1,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
                .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
                .initialLayout = vk::ImageLayout::eUndefined,
                .finalLayout = vk::ImageLayout::ePresentSrcKHR,
            },
            vk::AttachmentDescription{
                .format = vk::Format::eD32Sfloat,
                .samples = vk::SampleCountFlagBits::e1,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eDontCare,
                .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
                .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
                .initialLayout = vk::ImageLayout::eUndefined,
                .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            },
        };

        vk::AttachmentReference colorRef{
            .attachment = 0,
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
        };
        vk::AttachmentReference depthRef{
            .attachment = 1,
            .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        };

        vk::SubpassDescription subpass{
            .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorRef,
            .pDepthStencilAttachment = &depthRef,
        };

        vk::SubpassDependency dependency{
            .srcSubpass = std::numeric_limits<std::uint32_t>::max(),
            .dstSubpass = 0,
            .srcStageMask =
                vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
            .dstStageMask =
                vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
            .srcAccessMask = {},
            .dstAccessMask =
                vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite,
        };

        vk::RenderPassCreateInfo renderPassCreateInfo{
            .attachmentCount = static_cast<std::uint32_t>(attachments.size()),
            .pAttachments = attachments.data(),
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 1,
            .pDependencies = &dependency,
        };
        renderPass = device.createRenderPass(renderPassCreateInfo);
    }

    void createGraphicsPipeline() {
        const std::filesystem::path shaderDir{SHADER_SPV_DIR};
        std::vector<std::uint32_t> vertSpirv = readSpirvFile(shaderDir / "geometry.vert.spv");
        std::vector<std::uint32_t> geometrySpirv = readSpirvFile(shaderDir / "geometry.geom.spv");
        std::vector<std::uint32_t> fragSpirv = readSpirvFile(shaderDir / "geometry.frag.spv");

        vk::ShaderModuleCreateInfo vertModuleInfo{
            .codeSize = vertSpirv.size() * sizeof(std::uint32_t),
            .pCode = vertSpirv.data(),
        };
        vk::ShaderModuleCreateInfo geometryModuleInfo{
            .codeSize = geometrySpirv.size() * sizeof(std::uint32_t),
            .pCode = geometrySpirv.data(),
        };
        vk::ShaderModuleCreateInfo fragModuleInfo{
            .codeSize = fragSpirv.size() * sizeof(std::uint32_t),
            .pCode = fragSpirv.data(),
        };
        vertShaderModule = device.createShaderModule(vertModuleInfo);
        geometryShaderModule = device.createShaderModule(geometryModuleInfo);
        fragShaderModule = device.createShaderModule(fragModuleInfo);

        vk::PipelineShaderStageCreateInfo vertStage{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = *vertShaderModule,
            .pName = "main",
        };
        vk::PipelineShaderStageCreateInfo geometryStage{
            .stage = vk::ShaderStageFlagBits::eGeometry,
            .module = *geometryShaderModule,
            .pName = "main",
        };
        vk::PipelineShaderStageCreateInfo fragStage{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = *fragShaderModule,
            .pName = "main",
        };
        std::array<vk::PipelineShaderStageCreateInfo, 3> shaderStages = {vertStage, geometryStage, fragStage};

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::ePointList,
        };

        vk::Viewport viewport{
            .x = 0.f,
            .y = 0.f,
            .width = static_cast<float>(swapchainExtent.width),
            .height = static_cast<float>(swapchainExtent.height),
            .minDepth = 0.f,
            .maxDepth = 1.f,
        };
        vk::Rect2D scissor{.offset = {0, 0}, .extent = swapchainExtent};
        vk::PipelineViewportStateCreateInfo viewportState{
            .viewportCount = 1,
            .pViewports = &viewport,
            .scissorCount = 1,
            .pScissors = &scissor,
        };

        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eNone,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .lineWidth = 1.f,
        };

        vk::PipelineMultisampleStateCreateInfo multisampling{};

        vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = true,
            .depthWriteEnable = true,
            .depthCompareOp = vk::CompareOp::eLess,
        };

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
        };
        vk::PipelineColorBlendStateCreateInfo colorBlending{
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment,
        };

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo{
            .stageCount = static_cast<std::uint32_t>(shaderStages.size()),
            .pStages = shaderStages.data(),
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil,
            .pColorBlendState = &colorBlending,
            .layout = *pipelineLayout,
            .renderPass = *renderPass,
            .subpass = 0,
        };
        graphicsPipeline = device.createGraphicsPipeline(nullptr, pipelineInfo);
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

    void drawFrame() {
        const std::uint32_t frameIndex = currentFrame;

        (void)device.waitForFences({*inFlightFences[frameIndex]}, true, std::numeric_limits<std::uint64_t>::max());
        device.resetFences({*inFlightFences[frameIndex]});

        std::uint32_t imageIndex = 0;
        try {
            auto [acquireResult, idx] = swapchain.acquireNextImage(std::numeric_limits<std::uint64_t>::max(),
                                                                   *imageAvailableSemaphores[frameIndex], vk::Fence{});
            imageIndex = idx;
            (void)acquireResult;
        } catch (const vk::SystemError &err) {
            if (err.code().value() == static_cast<int>(vk::Result::eErrorOutOfDateKHR)) {
                return;
            }
            throw;
        }

        recordCommandBuffer(frameIndex, imageIndex);

        vk::PipelineStageFlags waitStages = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        vk::Semaphore waitSemaphore = *imageAvailableSemaphores[frameIndex];
        vk::Semaphore signalSemaphore = *renderFinishedSemaphores[imageIndex];
        vk::CommandBuffer commandBuffer = *commandBuffers[frameIndex];

        vk::SubmitInfo submitInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &waitSemaphore,
            .pWaitDstStageMask = &waitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &signalSemaphore,
        };
        graphicsQueue.submit(submitInfo, *inFlightFences[frameIndex]);

        vk::SwapchainKHR swapchainHandle = *swapchain;
        vk::PresentInfoKHR presentInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &signalSemaphore,
            .swapchainCount = 1,
            .pSwapchains = &swapchainHandle,
            .pImageIndices = &imageIndex,
        };
        try {
            (void)graphicsQueue.presentKHR(presentInfo);
        } catch (const vk::SystemError &err) {
            if (err.code().value() == static_cast<int>(vk::Result::eErrorOutOfDateKHR)) {
                return;
            }
            throw;
        }

        currentFrame = (currentFrame + 1) % maxConcurrentFrames;
    }

    void mainLoop() {
        while (!window.shouldClose()) {
            window.pollEvents();
            drawFrame();
        }
        device.waitIdle();
    }
};

int main() {
    try {
        GeometryShaderApplication app;
        app.run();
    } catch (const std::exception &e) {
        std::println("Error: {}", e.what());
        return 1;
    }

    return 0;
}
