export module common.windowed_app;

import std;
import vulkan;
import common.vulkan_instance;
import common.window;

export struct WindowedVulkanAppConfig {
    GlfwWindowConfig window;
    std::uint32_t apiVersion = vk::ApiVersion10;
    std::vector<std::string> requiredDeviceExtensions = {"VK_KHR_swapchain"};
    std::uint32_t framesInFlight = 2;
    vk::ImageUsageFlags swapchainImageUsage = vk::ImageUsageFlagBits::eColorAttachment;
    vk::PipelineStageFlags submitWaitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
};

export struct SwapchainAttachmentResource {
    vk::raii::DeviceMemory memory = nullptr;
    vk::raii::Image image = nullptr;
    vk::raii::ImageView imageView = nullptr;
};

export struct SwapchainRenderPassResources {
    // Members are ordered so Vulkan object destruction runs from framebuffers back to memory.
    std::vector<SwapchainAttachmentResource> ownedAttachments;
    std::vector<vk::raii::ImageView> swapchainImageViews;
    vk::raii::RenderPass renderPass = nullptr;
    std::vector<vk::raii::Framebuffer> framebuffers;
    std::vector<vk::ClearValue> defaultClearValues;
};

export class WindowedVulkanApp {
public:
    explicit WindowedVulkanApp(WindowedVulkanAppConfig config)
        : config_(validateConfig(std::move(config))),
          window_(config_.window),
          vulkan_({
              .applicationName = config_.window.title,
              .apiVersion = config_.apiVersion,
              .requiredExtensions = window_.requiredInstanceExtensions(),
          }),
          surface_(window_.createSurface(vulkan_.instance())) {}

    virtual ~WindowedVulkanApp() = default;

    WindowedVulkanApp(const WindowedVulkanApp &) = delete;
    WindowedVulkanApp &operator=(const WindowedVulkanApp &) = delete;
    WindowedVulkanApp(WindowedVulkanApp &&) = delete;
    WindowedVulkanApp &operator=(WindowedVulkanApp &&) = delete;

    void run() {
        if (hasRun_) {
            throw std::runtime_error("A WindowedVulkanApp instance can only be run once");
        }
        hasRun_ = true;

        try {
            initialize();
            mainLoop();
            device_.waitIdle();
        } catch (...) {
            waitIdleNoThrow();
            throw;
        }
    }

protected:
    virtual bool supportsDeviceFeatures(const vk::raii::PhysicalDevice &) const {
        return true;
    }

    virtual vk::PhysicalDeviceFeatures enabledDeviceFeatures() const {
        return {};
    }

    virtual vk::raii::Device createLogicalDevice(const vk::raii::PhysicalDevice &candidate,
                                                  const vk::DeviceQueueCreateInfo &queueCreateInfo) const {
        const auto features = enabledDeviceFeatures();

        std::vector<const char *> extensionNames;
        extensionNames.reserve(config_.requiredDeviceExtensions.size());
        for (const auto &extension : config_.requiredDeviceExtensions) {
            extensionNames.push_back(extension.c_str());
        }

        vk::DeviceCreateInfo createInfo{
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queueCreateInfo,
            .enabledExtensionCount = static_cast<std::uint32_t>(extensionNames.size()),
            .ppEnabledExtensionNames = extensionNames.data(),
            .pEnabledFeatures = &features,
        };
        return candidate.createDevice(createInfo);
    }

    virtual std::optional<SwapchainRenderPassResources> createSwapchainRenderPassResources() {
        if ((config_.swapchainImageUsage & vk::ImageUsageFlagBits::eColorAttachment) == vk::ImageUsageFlags{}) {
            throw std::runtime_error(
                "The default swapchain render pass requires eColorAttachment image usage; "
                "override createSwapchainRenderPassResources() to provide a custom path");
        }

        const auto depthFormat = findSupportedDepthFormat();
        const auto extent = swapchainExtent_;
        SwapchainRenderPassResources resources;
        resources.ownedAttachments.reserve(swapchainImages_.size());
        resources.swapchainImageViews.reserve(swapchainImages_.size());
        resources.framebuffers.reserve(swapchainImages_.size());

        for (const auto swapchainImage : swapchainImages_) {
            resources.swapchainImageViews.push_back(device_.createImageView({
                .image = swapchainImage,
                .viewType = vk::ImageViewType::e2D,
                .format = swapchainSurfaceFormat_.format,
                .subresourceRange =
                    {
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
            }));

            resources.ownedAttachments.emplace_back();
            auto &depthAttachment = resources.ownedAttachments.back();
            depthAttachment.image = device_.createImage({
                .imageType = vk::ImageType::e2D,
                .format = depthFormat,
                .extent = {extent.width, extent.height, 1},
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = vk::SampleCountFlagBits::e1,
                .tiling = vk::ImageTiling::eOptimal,
                .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
                .sharingMode = vk::SharingMode::eExclusive,
                .initialLayout = vk::ImageLayout::eUndefined,
            });

            const auto memoryRequirements = depthAttachment.image.getMemoryRequirements();
            depthAttachment.memory = vk::raii::DeviceMemory(
                device_,
                {
                    .allocationSize = memoryRequirements.size,
                    .memoryTypeIndex = findMemoryTypeIndex(memoryRequirements,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal),
                });
            depthAttachment.image.bindMemory(*depthAttachment.memory, 0);

            depthAttachment.imageView = device_.createImageView({
                .image = *depthAttachment.image,
                .viewType = vk::ImageViewType::e2D,
                .format = depthFormat,
                .subresourceRange =
                    {
                        .aspectMask = depthAspectMask(depthFormat),
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
            });
        }

        std::array attachments = {
            vk::AttachmentDescription{
                .format = swapchainSurfaceFormat_.format,
                .samples = vk::SampleCountFlagBits::e1,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
                .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
                .initialLayout = vk::ImageLayout::eUndefined,
                .finalLayout = vk::ImageLayout::ePresentSrcKHR,
            },
            vk::AttachmentDescription{
                .format = depthFormat,
                .samples = vk::SampleCountFlagBits::e1,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eDontCare,
                .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
                .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
                .initialLayout = vk::ImageLayout::eUndefined,
                .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            },
        };
        vk::AttachmentReference colorAttachment{
            .attachment = 0,
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
        };
        vk::AttachmentReference depthAttachment{
            .attachment = 1,
            .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        };
        vk::SubpassDescription subpass{
            .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachment,
            .pDepthStencilAttachment = &depthAttachment,
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
        resources.renderPass = device_.createRenderPass({
            .attachmentCount = static_cast<std::uint32_t>(attachments.size()),
            .pAttachments = attachments.data(),
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 1,
            .pDependencies = &dependency,
        });

        for (std::size_t index = 0; index < swapchainImages_.size(); ++index) {
            const std::array framebufferAttachments = {
                *resources.swapchainImageViews[index],
                *resources.ownedAttachments[index].imageView,
            };
            resources.framebuffers.push_back(device_.createFramebuffer({
                .renderPass = *resources.renderPass,
                .attachmentCount = static_cast<std::uint32_t>(framebufferAttachments.size()),
                .pAttachments = framebufferAttachments.data(),
                .width = extent.width,
                .height = extent.height,
                .layers = 1,
            }));
        }

        resources.defaultClearValues = {vk::ClearValue{}, vk::ClearValue{}};
        resources.defaultClearValues[0].color.setFloat32(std::array<float, 4>{0.02f, 0.02f, 0.08f, 1.0f});
        resources.defaultClearValues[1].depthStencil = vk::ClearDepthStencilValue{1.0f, 0};
        return resources;
    }

    virtual void createApplicationResources() = 0;
    virtual void recordFrameCommands(vk::raii::CommandBuffer &commandBuffer, std::uint32_t imageIndex) = 0;

    [[nodiscard]] const vk::raii::PhysicalDevice &physicalDevice() const noexcept {
        return physicalDevice_;
    }

    [[nodiscard]] const vk::raii::Device &device() const noexcept {
        return device_;
    }

    [[nodiscard]] const vk::raii::SurfaceKHR &surface() const noexcept {
        return surface_;
    }

    [[nodiscard]] const vk::raii::Queue &graphicsQueue() const noexcept {
        return graphicsQueue_;
    }

    [[nodiscard]] const vk::raii::CommandPool &commandPool() const noexcept {
        return commandPool_;
    }

    [[nodiscard]] std::uint32_t graphicsQueueFamilyIndex() const noexcept {
        return graphicsQueueFamilyIndex_;
    }

    [[nodiscard]] GlfwWindow &window() noexcept {
        return window_;
    }

    [[nodiscard]] const GlfwWindow &window() const noexcept {
        return window_;
    }

    [[nodiscard]] vk::SurfaceFormatKHR swapchainSurfaceFormat() const noexcept {
        return swapchainSurfaceFormat_;
    }

    [[nodiscard]] vk::Extent2D swapchainExtent() const noexcept {
        return swapchainExtent_;
    }

    [[nodiscard]] const std::vector<vk::Image> &swapchainImages() const noexcept {
        return swapchainImages_;
    }

    [[nodiscard]] const std::vector<std::string> &requiredDeviceExtensions() const noexcept {
        return config_.requiredDeviceExtensions;
    }

    [[nodiscard]] bool hasSwapchainRenderPass() const noexcept {
        return swapchainRenderPassResources_.has_value();
    }

    [[nodiscard]] vk::RenderPass swapchainRenderPass() const {
        return *requireSwapchainRenderPassResources().renderPass;
    }

    [[nodiscard]] vk::Framebuffer swapchainFramebuffer(std::uint32_t imageIndex) const {
        const auto &resources = requireSwapchainRenderPassResources();
        if (imageIndex >= resources.framebuffers.size()) {
            throw std::out_of_range("Swapchain framebuffer index is out of range");
        }
        return *resources.framebuffers[imageIndex];
    }

    void beginSwapchainRenderPass(vk::raii::CommandBuffer &commandBuffer, std::uint32_t imageIndex,
                                  vk::SubpassContents contents = vk::SubpassContents::eInline) const {
        const auto &clearValues = requireSwapchainRenderPassResources().defaultClearValues;
        beginSwapchainRenderPass(commandBuffer, imageIndex,
                                 std::span<const vk::ClearValue>{clearValues.data(), clearValues.size()}, contents);
    }

    void beginSwapchainRenderPass(vk::raii::CommandBuffer &commandBuffer, std::uint32_t imageIndex,
                                  std::span<const vk::ClearValue> clearValues,
                                  vk::SubpassContents contents = vk::SubpassContents::eInline) const {
        vk::RenderPassBeginInfo beginInfo{
            .renderPass = swapchainRenderPass(),
            .framebuffer = swapchainFramebuffer(imageIndex),
            .renderArea = {.offset = {0, 0}, .extent = swapchainExtent_},
            .clearValueCount = static_cast<std::uint32_t>(clearValues.size()),
            .pClearValues = clearValues.data(),
        };
        commandBuffer.beginRenderPass(beginInfo, contents);
    }

    [[nodiscard]] std::uint32_t findMemoryTypeIndex(const vk::MemoryRequirements &memoryRequirements,
                                                    vk::MemoryPropertyFlags properties) const {
        const auto memoryProperties = physicalDevice_.getMemoryProperties();
        for (std::uint32_t index = 0; index < memoryProperties.memoryTypeCount; ++index) {
            if ((memoryRequirements.memoryTypeBits & (1u << index)) != 0 &&
                (memoryProperties.memoryTypes[index].propertyFlags & properties) == properties) {
                return index;
            }
        }
        throw std::runtime_error(std::format("No compatible memory type for properties {}",
                                             vk::to_string(properties)));
    }

private:
    static WindowedVulkanAppConfig validateConfig(WindowedVulkanAppConfig config) {
        if (config.framesInFlight == 0) {
            throw std::runtime_error("framesInFlight must be greater than zero");
        }
        if (!config.swapchainImageUsage) {
            throw std::runtime_error("swapchainImageUsage must not be empty");
        }
        if (!config.submitWaitStage) {
            throw std::runtime_error("submitWaitStage must not be empty");
        }
        if (std::find(config.requiredDeviceExtensions.begin(), config.requiredDeviceExtensions.end(),
                      "VK_KHR_swapchain") == config.requiredDeviceExtensions.end()) {
            config.requiredDeviceExtensions.emplace_back("VK_KHR_swapchain");
        }
        return config;
    }

    static bool hasExtension(const std::vector<vk::ExtensionProperties> &properties, std::string_view name) {
        return std::ranges::any_of(properties, [name](const auto &property) {
            return name == property.extensionName.data();
        });
    }

    [[nodiscard]] bool supportsRequiredExtensions(const vk::raii::PhysicalDevice &candidate) const {
        const auto properties = candidate.enumerateDeviceExtensionProperties();
        return std::ranges::all_of(config_.requiredDeviceExtensions, [&](const auto &extension) {
            return hasExtension(properties, extension);
        });
    }

    [[nodiscard]] std::optional<std::uint32_t>
    findGraphicsPresentQueue(const vk::raii::PhysicalDevice &candidate) const {
        const auto queueFamilies = candidate.getQueueFamilyProperties();
        for (std::uint32_t index = 0; index < queueFamilies.size(); ++index) {
            if ((queueFamilies[index].queueFlags & vk::QueueFlagBits::eGraphics) &&
                candidate.getSurfaceSupportKHR(index, *surface_)) {
                return index;
            }
        }
        return std::nullopt;
    }

    void initialize() {
        selectPhysicalDevice();
        createDevice();
        createCommandPool();
        createSwapchain();
        swapchainRenderPassResources_ = createSwapchainRenderPassResources();
        validateSwapchainRenderPassResources();
        createApplicationResources();
        createCommandBuffers();
        createSyncObjects();
    }

    void selectPhysicalDevice() {
        const auto candidates = vulkan_.instance().enumeratePhysicalDevices();
        if (candidates.empty()) {
            throw std::runtime_error("Failed to find GPUs with Vulkan support");
        }

        for (const auto &candidate : candidates) {
            const auto properties = candidate.getProperties();
            std::println("Physical Device: {}, version:{}.{}.{}, ", properties.deviceName.data(),
                         vk::apiVersionMajor(properties.apiVersion), vk::apiVersionMinor(properties.apiVersion),
                         vk::apiVersionPatch(properties.apiVersion));
        }

        for (const auto &candidate : candidates) {
            if (!supportsRequiredExtensions(candidate) || !supportsDeviceFeatures(candidate)) {
                continue;
            }
            const auto queueFamily = findGraphicsPresentQueue(candidate);
            if (!queueFamily) {
                continue;
            }

            physicalDevice_ = candidate;
            graphicsQueueFamilyIndex_ = *queueFamily;
            std::println("Selected GPU: {}", candidate.getProperties().deviceName.data());
            return;
        }

        throw std::runtime_error(std::format(
            "No suitable GPU found for {} (required device extensions, features, and graphics/present support)",
            config_.window.title));
    }

    void createDevice() {
        const float priority = 1.0f;
        vk::DeviceQueueCreateInfo queueCreateInfo{
            .queueFamilyIndex = graphicsQueueFamilyIndex_,
            .queueCount = 1,
            .pQueuePriorities = &priority,
        };
        device_ = createLogicalDevice(physicalDevice_, queueCreateInfo);
        graphicsQueue_ = device_.getQueue(graphicsQueueFamilyIndex_, 0);
    }

    void createCommandPool() {
        vk::CommandPoolCreateInfo createInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = graphicsQueueFamilyIndex_,
        };
        commandPool_ = device_.createCommandPool(createInfo);
    }

protected:
    static vk::SurfaceFormatKHR chooseSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &formats) {
        for (const auto &format : formats) {
            if (format.format == vk::Format::eB8G8R8A8Srgb &&
                format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return format;
            }
        }
        return formats.front();
    }

private:
    static vk::PresentModeKHR choosePresentMode(const std::vector<vk::PresentModeKHR> &presentModes) {
        if (std::ranges::find(presentModes, vk::PresentModeKHR::eMailbox) != presentModes.end()) {
            return vk::PresentModeKHR::eMailbox;
        }
        return vk::PresentModeKHR::eFifo;
    }

    static vk::Extent2D chooseExtent(const vk::SurfaceCapabilitiesKHR &capabilities, vk::Extent2D framebufferExtent) {
        if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        return {
            .width = std::clamp(framebufferExtent.width, capabilities.minImageExtent.width,
                               capabilities.maxImageExtent.width),
            .height = std::clamp(framebufferExtent.height, capabilities.minImageExtent.height,
                                capabilities.maxImageExtent.height),
        };
    }

    static vk::CompositeAlphaFlagBitsKHR chooseCompositeAlpha(vk::CompositeAlphaFlagsKHR supported) {
        constexpr std::array candidates = {
            vk::CompositeAlphaFlagBitsKHR::eOpaque,
            vk::CompositeAlphaFlagBitsKHR::ePreMultiplied,
            vk::CompositeAlphaFlagBitsKHR::ePostMultiplied,
            vk::CompositeAlphaFlagBitsKHR::eInherit,
        };
        for (const auto candidate : candidates) {
            if (supported & candidate) {
                return candidate;
            }
        }
        throw std::runtime_error("Surface does not report a supported composite alpha mode");
    }

    [[nodiscard]] vk::Format findSupportedDepthFormat() const {
        constexpr std::array candidates = {
            vk::Format::eD32Sfloat,
            vk::Format::eD32SfloatS8Uint,
            vk::Format::eD24UnormS8Uint,
            vk::Format::eD16Unorm,
        };
        for (const auto format : candidates) {
            if (physicalDevice_.getFormatProperties(format).optimalTilingFeatures &
                vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
                return format;
            }
        }
        throw std::runtime_error("No supported depth-stencil attachment format is available");
    }

    static vk::ImageAspectFlags depthAspectMask(vk::Format format) {
        switch (format) {
        case vk::Format::eD32SfloatS8Uint:
        case vk::Format::eD24UnormS8Uint:
            return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
        default:
            return vk::ImageAspectFlagBits::eDepth;
        }
    }

    [[nodiscard]] const SwapchainRenderPassResources &requireSwapchainRenderPassResources() const {
        if (!swapchainRenderPassResources_) {
            throw std::logic_error("This application did not create swapchain render pass resources");
        }
        return *swapchainRenderPassResources_;
    }

    void validateSwapchainRenderPassResources() const {
        if (!swapchainRenderPassResources_) {
            return;
        }

        const auto &resources = *swapchainRenderPassResources_;
        if (*resources.renderPass == vk::RenderPass{}) {
            throw std::runtime_error("Swapchain render pass resources must contain a render pass");
        }
        if (resources.framebuffers.size() != swapchainImages_.size()) {
            throw std::runtime_error("Swapchain render pass resources must provide one framebuffer per swapchain image");
        }
    }

    void createSwapchain() {
        const auto capabilities = physicalDevice_.getSurfaceCapabilitiesKHR(*surface_);
        const auto formats = physicalDevice_.getSurfaceFormatsKHR(*surface_);
        const auto presentModes = physicalDevice_.getSurfacePresentModesKHR(*surface_);
        if (formats.empty()) {
            throw std::runtime_error("No surface formats are available");
        }
        if (presentModes.empty()) {
            throw std::runtime_error("No present modes are available");
        }
        if ((capabilities.supportedUsageFlags & config_.swapchainImageUsage) != config_.swapchainImageUsage) {
            throw std::runtime_error("The window surface does not support the requested swapchain image usage");
        }

        swapchainSurfaceFormat_ = chooseSurfaceFormat(formats);
        const auto presentMode = choosePresentMode(presentModes);
        swapchainExtent_ = chooseExtent(capabilities, window_.framebufferExtent());

        std::uint32_t imageCount = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
            imageCount = capabilities.maxImageCount;
        }

        vk::SwapchainCreateInfoKHR createInfo{
            .surface = *surface_,
            .minImageCount = imageCount,
            .imageFormat = swapchainSurfaceFormat_.format,
            .imageColorSpace = swapchainSurfaceFormat_.colorSpace,
            .imageExtent = swapchainExtent_,
            .imageArrayLayers = 1,
            .imageUsage = config_.swapchainImageUsage,
            .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform = capabilities.currentTransform,
            .compositeAlpha = chooseCompositeAlpha(capabilities.supportedCompositeAlpha),
            .presentMode = presentMode,
            .clipped = true,
        };
        swapchain_ = device_.createSwapchainKHR(createInfo);
        swapchainImages_ = swapchain_.getImages();

        std::println("Swapchain: {} images, extent {}x{}, format {}, present mode {}", swapchainImages_.size(),
                     swapchainExtent_.width, swapchainExtent_.height,
                     vk::to_string(swapchainSurfaceFormat_.format), vk::to_string(presentMode));
    }

    void createCommandBuffers() {
        vk::CommandBufferAllocateInfo allocateInfo{
            .commandPool = *commandPool_,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = config_.framesInFlight,
        };
        commandBuffers_ = device_.allocateCommandBuffers(allocateInfo);
    }

    void createSyncObjects() {
        imageAvailableSemaphores_.reserve(config_.framesInFlight);
        inFlightFences_.reserve(config_.framesInFlight);
        renderFinishedSemaphores_.reserve(swapchainImages_.size());

        for (std::uint32_t index = 0; index < config_.framesInFlight; ++index) {
            imageAvailableSemaphores_.push_back(device_.createSemaphore({}));
            inFlightFences_.push_back(device_.createFence({.flags = vk::FenceCreateFlagBits::eSignaled}));
        }
        for (std::size_t index = 0; index < swapchainImages_.size(); ++index) {
            renderFinishedSemaphores_.push_back(device_.createSemaphore({}));
        }
    }

    static bool isSwapchainOutOfDate(const vk::SystemError &error) {
        return error.code().value() == static_cast<int>(vk::Result::eErrorOutOfDateKHR);
    }

    void drawFrame() {
        const auto frameIndex = currentFrame_;
        (void)device_.waitForFences({*inFlightFences_[frameIndex]}, true,
                                    std::numeric_limits<std::uint64_t>::max());

        std::uint32_t imageIndex = 0;
        try {
            const auto [result, acquiredImageIndex] = swapchain_.acquireNextImage(
                std::numeric_limits<std::uint64_t>::max(), *imageAvailableSemaphores_[frameIndex], vk::Fence{});
            (void)result;
            imageIndex = acquiredImageIndex;
        } catch (const vk::SystemError &error) {
            if (isSwapchainOutOfDate(error)) {
                throw std::runtime_error("Swapchain became out of date; this fixed-size example does not recreate it");
            }
            throw;
        }

        device_.resetFences({*inFlightFences_[frameIndex]});
        auto &commandBuffer = commandBuffers_[frameIndex];
        commandBuffer.reset({});
        commandBuffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        recordFrameCommands(commandBuffer, imageIndex);
        commandBuffer.end();

        const vk::Semaphore waitSemaphore = *imageAvailableSemaphores_[frameIndex];
        const vk::CommandBuffer commandBufferHandle = *commandBuffer;
        const vk::Semaphore signalSemaphore = *renderFinishedSemaphores_[imageIndex];
        vk::SubmitInfo submitInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &waitSemaphore,
            .pWaitDstStageMask = &config_.submitWaitStage,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBufferHandle,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &signalSemaphore,
        };
        graphicsQueue_.submit(submitInfo, *inFlightFences_[frameIndex]);

        const vk::SwapchainKHR swapchainHandle = *swapchain_;
        vk::PresentInfoKHR presentInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &signalSemaphore,
            .swapchainCount = 1,
            .pSwapchains = &swapchainHandle,
            .pImageIndices = &imageIndex,
        };
        try {
            (void)graphicsQueue_.presentKHR(presentInfo);
        } catch (const vk::SystemError &error) {
            if (isSwapchainOutOfDate(error)) {
                throw std::runtime_error("Swapchain became out of date; this fixed-size example does not recreate it");
            }
            throw;
        }

        currentFrame_ = (currentFrame_ + 1) % config_.framesInFlight;
    }

    void mainLoop() {
        while (!window_.shouldClose()) {
            window_.pollEvents();
            drawFrame();
        }
    }

    void waitIdleNoThrow() noexcept {
        if (*device_) {
            try {
                device_.waitIdle();
            } catch (...) {
            }
        }
    }

    WindowedVulkanAppConfig config_;
    GlfwWindow window_;
    VulkanInstance vulkan_;
    vk::raii::SurfaceKHR surface_ = nullptr;
    vk::raii::PhysicalDevice physicalDevice_ = nullptr;
    vk::raii::Device device_ = nullptr;
    vk::raii::Queue graphicsQueue_ = nullptr;
    std::uint32_t graphicsQueueFamilyIndex_ = 0;
    vk::raii::CommandPool commandPool_ = nullptr;
    vk::raii::SwapchainKHR swapchain_ = nullptr;
    vk::SurfaceFormatKHR swapchainSurfaceFormat_{};
    vk::Extent2D swapchainExtent_{};
    std::vector<vk::Image> swapchainImages_;
    std::optional<SwapchainRenderPassResources> swapchainRenderPassResources_;
    std::vector<vk::raii::CommandBuffer> commandBuffers_;
    std::vector<vk::raii::Semaphore> imageAvailableSemaphores_;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores_;
    std::vector<vk::raii::Fence> inFlightFences_;
    std::uint32_t currentFrame_ = 0;
    bool hasRun_ = false;
};
