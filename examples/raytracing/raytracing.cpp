import vulkan;
import std;
import common;

#ifndef SHADER_SPV_DIR
#error "SHADER_SPV_DIR must be set by CMake (see examples/raytracing/CMakeLists.txt)"
#endif

namespace {
constexpr std::uint32_t maxConcurrentFrames = 2;
constexpr vk::Format outputFormat = vk::Format::eR8G8B8A8Unorm;

template <typename T>
constexpr T alignUp(T value, T alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

struct BufferResource {
    vk::raii::DeviceMemory memory = nullptr;
    vk::raii::Buffer buffer = nullptr;
    vk::DeviceSize size = 0;
};

struct ImageResource {
    vk::raii::DeviceMemory memory = nullptr;
    vk::raii::Image image = nullptr;
    vk::raii::ImageView view = nullptr;
};

struct AccelerationStructureResource {
    BufferResource storage;
    vk::raii::AccelerationStructureKHR handle = nullptr;
};

struct Vertex {
    float x;
    float y;
    float z;
};

}  // namespace

class RayTracingApplication {
public:
    RayTracingApplication()
        : window({
              .width = 1024,
              .height = 768,
              .title = "Vulkan Ray Tracing Triangle",
          }),
          vulkan({
              .applicationName = "Vulkan Ray Tracing Triangle",
              .apiVersion = vk::ApiVersion12,
              .requiredExtensions = window.requiredInstanceExtensions(),
          }),
          surface(window.createSurface(vulkan.instance())) {}

    ~RayTracingApplication() {
        if (*device) {
            try {
                device.waitIdle();
            } catch (...) {
            }
        }
    }

    void run() {
        initVulkan();
        mainLoop();
    }

private:
    static constexpr std::array<const char *, 4> requiredDeviceExtensions = {
        "VK_KHR_swapchain",
        "VK_KHR_deferred_host_operations",
        "VK_KHR_acceleration_structure",
        "VK_KHR_ray_tracing_pipeline",
    };

    GlfwWindow window;
    VulkanInstance vulkan;
    vk::raii::SurfaceKHR surface = nullptr;
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;
    vk::raii::Queue graphicsQueue = nullptr;
    std::uint32_t graphicsQueueFamilyIndex = 0;

    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingProperties{};
    vk::PhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructureProperties{};

    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;

    vk::raii::SwapchainKHR swapchain = nullptr;
    vk::SurfaceFormatKHR swapchainSurfaceFormat{};
    vk::Extent2D swapchainExtent{};
    std::vector<vk::Image> swapchainImages;
    ImageResource outputImage;

    BufferResource vertexBuffer;
    BufferResource indexBuffer;
    AccelerationStructureResource bottomLevelAS;
    BufferResource instanceBuffer;
    AccelerationStructureResource topLevelAS;

    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> descriptorSets;

    std::vector<vk::raii::ShaderModule> shaderModules;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline rayTracingPipeline = nullptr;
    BufferResource shaderBindingTable;
    vk::StridedDeviceAddressRegionKHR raygenRegion{};
    vk::StridedDeviceAddressRegionKHR missRegion{};
    vk::StridedDeviceAddressRegionKHR hitRegion{};
    vk::StridedDeviceAddressRegionKHR callableRegion{};

    std::vector<vk::raii::Semaphore> imageAvailableSemaphores;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;
    std::uint32_t currentFrame = 0;

    void initVulkan() {
        selectPhysicalDevice();
        createDevice();
        createCommandPool();
        createSwapchain();
        createOutputImage();
        createGeometryBuffers();
        createAccelerationStructures();
        createDescriptors();
        createRayTracingPipeline();
        createShaderBindingTable();
        createCommandBuffers();
        createSyncObjects();
    }

    static bool hasExtension(const std::vector<vk::ExtensionProperties> &properties, std::string_view name) {
        for (const auto &property : properties) {
            if (name == property.extensionName.data()) {
                return true;
            }
        }
        return false;
    }

    static std::optional<std::uint32_t> findGraphicsPresentQueue(const vk::raii::PhysicalDevice &candidate,
                                                                 vk::SurfaceKHR candidateSurface) {
        const auto queueFamilies = candidate.getQueueFamilyProperties();
        for (std::uint32_t index = 0; index < queueFamilies.size(); ++index) {
            if (!(queueFamilies[index].queueFlags & vk::QueueFlagBits::eGraphics)) {
                continue;
            }
            if (candidate.getSurfaceSupportKHR(index, candidateSurface)) {
                return index;
            }
        }
        return std::nullopt;
    }

    static bool supportsRequiredExtensions(const vk::raii::PhysicalDevice &candidate) {
        const auto available = candidate.enumerateDeviceExtensionProperties();
        for (const char *required : requiredDeviceExtensions) {
            if (!hasExtension(available, required)) {
                return false;
            }
        }
        return true;
    }

    void selectPhysicalDevice() {
        const auto candidates = vulkan.instance().enumeratePhysicalDevices();
        if (candidates.empty()) {
            throw std::runtime_error("No Vulkan physical devices were found");
        }

        for (const auto &candidate : candidates) {
            const auto properties = candidate.getProperties();
            std::println("Physical device: {}", properties.deviceName.data());

            if (properties.apiVersion < vk::ApiVersion12 || !supportsRequiredExtensions(candidate)) {
                continue;
            }

            const auto queueFamily = findGraphicsPresentQueue(candidate, *surface);
            if (!queueFamily.has_value()) {
                continue;
            }

            const auto features = candidate.getFeatures2<vk::PhysicalDeviceFeatures2,
                                                         vk::PhysicalDeviceBufferDeviceAddressFeatures,
                                                         vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
                                                         vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>();
            if (!features.get<vk::PhysicalDeviceBufferDeviceAddressFeatures>().bufferDeviceAddress ||
                !features.get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>().accelerationStructure ||
                !features.get<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>().rayTracingPipeline) {
                continue;
            }

            const auto surfaceCapabilities = candidate.getSurfaceCapabilitiesKHR(*surface);
            if (!(surfaceCapabilities.supportedUsageFlags & vk::ImageUsageFlagBits::eTransferDst)) {
                continue;
            }

            const auto outputProperties = candidate.getFormatProperties(outputFormat);
            const auto requiredOutputFeatures =
                vk::FormatFeatureFlagBits::eStorageImage | vk::FormatFeatureFlagBits::eBlitSrc;
            if ((outputProperties.optimalTilingFeatures & requiredOutputFeatures) != requiredOutputFeatures) {
                continue;
            }

            physicalDevice = candidate;
            graphicsQueueFamilyIndex = *queueFamily;

            const auto extendedProperties =
                candidate.getProperties2<vk::PhysicalDeviceProperties2,
                                         vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
                                         vk::PhysicalDeviceAccelerationStructurePropertiesKHR>();
            rayTracingProperties = extendedProperties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
            accelerationStructureProperties =
                extendedProperties.get<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>();

            if (rayTracingProperties.maxRayRecursionDepth < 1) {
                continue;
            }

            std::println("Selected ray tracing GPU: {}", properties.deviceName.data());
            return;
        }

        throw std::runtime_error(
            "No GPU supports graphics/present, Vulkan 1.2, buffer device address, acceleration structures, and the KHR ray tracing pipeline");
    }

    void createDevice() {
        const float queuePriority = 1.0f;
        vk::DeviceQueueCreateInfo queueCreateInfo{
            .queueFamilyIndex = graphicsQueueFamilyIndex,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority,
        };

        vk::PhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{
            .bufferDeviceAddress = true,
        };
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{
            .pNext = &bufferDeviceAddressFeatures,
            .accelerationStructure = true,
        };
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{
            .pNext = &accelerationStructureFeatures,
            .rayTracingPipeline = true,
        };

        vk::DeviceCreateInfo createInfo{
            .pNext = &rayTracingPipelineFeatures,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queueCreateInfo,
            .enabledExtensionCount = static_cast<std::uint32_t>(requiredDeviceExtensions.size()),
            .ppEnabledExtensionNames = requiredDeviceExtensions.data(),
        };

        device = physicalDevice.createDevice(createInfo);
        graphicsQueue = device.getQueue(graphicsQueueFamilyIndex, 0);
    }

    void createCommandPool() {
        vk::CommandPoolCreateInfo createInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = graphicsQueueFamilyIndex,
        };
        commandPool = device.createCommandPool(createInfo);
    }

    static vk::SurfaceFormatKHR chooseSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &formats) {
        for (const auto &format : formats) {
            if (format.format == vk::Format::eB8G8R8A8Srgb &&
                format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return format;
            }
        }
        return formats.front();
    }

    static vk::PresentModeKHR choosePresentMode(const std::vector<vk::PresentModeKHR> &presentModes) {
        for (const auto mode : presentModes) {
            if (mode == vk::PresentModeKHR::eMailbox) {
                return mode;
            }
        }
        return vk::PresentModeKHR::eFifo;
    }

    static vk::Extent2D chooseExtent(const vk::SurfaceCapabilitiesKHR &capabilities, int width, int height) {
        if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
            return capabilities.currentExtent;
        }

        vk::Extent2D extent{
            static_cast<std::uint32_t>(width),
            static_cast<std::uint32_t>(height),
        };
        extent.width = std::clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        extent.height =
            std::clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
        return extent;
    }

    void createSwapchain() {
        const auto capabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
        const auto formats = physicalDevice.getSurfaceFormatsKHR(*surface);
        const auto presentModes = physicalDevice.getSurfacePresentModesKHR(*surface);
        if (formats.empty() || presentModes.empty()) {
            throw std::runtime_error("The selected GPU has no usable swapchain formats or present modes");
        }
        if (!(capabilities.supportedUsageFlags & vk::ImageUsageFlagBits::eTransferDst)) {
            throw std::runtime_error("The window surface does not support transfer-destination swapchain images");
        }

        swapchainSurfaceFormat = chooseSurfaceFormat(formats);
        const auto destinationFormatProperties = physicalDevice.getFormatProperties(swapchainSurfaceFormat.format);
        if (!(destinationFormatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eBlitDst)) {
            throw std::runtime_error("The selected swapchain format does not support blit destination operations");
        }

        const auto framebufferExtent = window.framebufferExtent();
        swapchainExtent = chooseExtent(capabilities, static_cast<int>(framebufferExtent.width),
                                       static_cast<int>(framebufferExtent.height));

        std::uint32_t imageCount = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
            imageCount = capabilities.maxImageCount;
        }

        auto compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
        if (!(capabilities.supportedCompositeAlpha & compositeAlpha)) {
            if (capabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::ePreMultiplied) {
                compositeAlpha = vk::CompositeAlphaFlagBitsKHR::ePreMultiplied;
            } else if (capabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::ePostMultiplied) {
                compositeAlpha = vk::CompositeAlphaFlagBitsKHR::ePostMultiplied;
            } else {
                compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eInherit;
            }
        }

        const auto presentMode = choosePresentMode(presentModes);
        vk::SwapchainCreateInfoKHR createInfo{
            .surface = *surface,
            .minImageCount = imageCount,
            .imageFormat = swapchainSurfaceFormat.format,
            .imageColorSpace = swapchainSurfaceFormat.colorSpace,
            .imageExtent = swapchainExtent,
            .imageArrayLayers = 1,
            .imageUsage = vk::ImageUsageFlagBits::eTransferDst,
            .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform = capabilities.currentTransform,
            .compositeAlpha = compositeAlpha,
            .presentMode = presentMode,
            .clipped = true,
        };

        swapchain = device.createSwapchainKHR(createInfo);
        swapchainImages = swapchain.getImages();
        std::println("Swapchain: {} images, {}x{}, {}, {}", swapchainImages.size(), swapchainExtent.width,
                     swapchainExtent.height, vk::to_string(swapchainSurfaceFormat.format), vk::to_string(presentMode));
    }

    std::uint32_t findMemoryType(std::uint32_t typeBits, vk::MemoryPropertyFlags requiredProperties) const {
        const auto memoryProperties = physicalDevice.getMemoryProperties();
        for (std::uint32_t index = 0; index < memoryProperties.memoryTypeCount; ++index) {
            if ((typeBits & (1u << index)) != 0 &&
                (memoryProperties.memoryTypes[index].propertyFlags & requiredProperties) == requiredProperties) {
                return index;
            }
        }
        throw std::runtime_error(std::format("No compatible memory type for properties {}",
                                             vk::to_string(requiredProperties)));
    }

    BufferResource createBuffer(vk::DeviceSize size,
                                vk::BufferUsageFlags usage,
                                vk::MemoryPropertyFlags memoryProperties,
                                bool needsDeviceAddress) {
        BufferResource resource;
        resource.size = size;

        vk::BufferCreateInfo bufferCreateInfo{
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
        };
        resource.buffer = device.createBuffer(bufferCreateInfo);

        const auto requirements = resource.buffer.getMemoryRequirements();
        vk::MemoryAllocateFlagsInfo allocateFlagsInfo{
            .flags = needsDeviceAddress ? vk::MemoryAllocateFlagBits::eDeviceAddress : vk::MemoryAllocateFlags{},
        };
        vk::MemoryAllocateInfo allocateInfo{
            .pNext = needsDeviceAddress ? &allocateFlagsInfo : nullptr,
            .allocationSize = requirements.size,
            .memoryTypeIndex = findMemoryType(requirements.memoryTypeBits, memoryProperties),
        };
        resource.memory = vk::raii::DeviceMemory(device, allocateInfo);
        resource.buffer.bindMemory(*resource.memory, 0);
        return resource;
    }

    void uploadBuffer(const BufferResource &resource, const void *data, vk::DeviceSize size) {
        void *mapped = resource.memory.mapMemory(0, size);
        std::memcpy(mapped, data, static_cast<std::size_t>(size));
        resource.memory.unmapMemory();
    }

    vk::DeviceAddress getBufferAddress(const BufferResource &resource) const {
        vk::BufferDeviceAddressInfo info{.buffer = *resource.buffer};
        return device.getBufferAddress(info);
    }

    static vk::ImageSubresourceRange colorSubresourceRange() {
        return vk::ImageSubresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };
    }

    template <typename Recorder>
    void submitImmediate(Recorder &&recorder) {
        vk::CommandBufferAllocateInfo allocateInfo{
            .commandPool = *commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        };
        auto buffers = device.allocateCommandBuffers(allocateInfo);
        auto &commandBuffer = buffers.front();

        commandBuffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        std::invoke(std::forward<Recorder>(recorder), commandBuffer);
        commandBuffer.end();

        const vk::CommandBuffer rawCommandBuffer = *commandBuffer;
        vk::SubmitInfo submitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &rawCommandBuffer,
        };
        graphicsQueue.submit(submitInfo);
        graphicsQueue.waitIdle();
    }

    void createOutputImage() {
        vk::ImageCreateInfo imageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = outputFormat,
            .extent = {swapchainExtent.width, swapchainExtent.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined,
        };
        outputImage.image = device.createImage(imageCreateInfo);

        const auto requirements = outputImage.image.getMemoryRequirements();
        vk::MemoryAllocateInfo allocateInfo{
            .allocationSize = requirements.size,
            .memoryTypeIndex =
                findMemoryType(requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
        };
        outputImage.memory = vk::raii::DeviceMemory(device, allocateInfo);
        outputImage.image.bindMemory(*outputImage.memory, 0);

        vk::ImageViewCreateInfo viewCreateInfo{
            .image = *outputImage.image,
            .viewType = vk::ImageViewType::e2D,
            .format = outputFormat,
            .subresourceRange = colorSubresourceRange(),
        };
        outputImage.view = device.createImageView(viewCreateInfo);

        submitImmediate([this](vk::raii::CommandBuffer &commandBuffer) {
            vk::ImageMemoryBarrier barrier{
                .srcAccessMask = {},
                .dstAccessMask = vk::AccessFlagBits::eShaderWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                .image = *outputImage.image,
                .subresourceRange = colorSubresourceRange(),
            };
            commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                                          vk::PipelineStageFlagBits::eRayTracingShaderKHR, {}, {}, {}, {barrier});
        });
    }

    void createGeometryBuffers() {
        constexpr std::array<Vertex, 3> vertices = {
            Vertex{-0.8f, 0.6f, 0.0f},
            Vertex{0.8f, 0.6f, 0.0f},
            Vertex{0.0f, -0.7f, 0.0f},
        };
        constexpr std::array<std::uint32_t, 3> indices = {0, 1, 2};

        constexpr auto hostMemory =
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
        const auto buildInputUsage = vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
                                     vk::BufferUsageFlagBits::eShaderDeviceAddress;

        vertexBuffer = createBuffer(sizeof(vertices), buildInputUsage, hostMemory, true);
        uploadBuffer(vertexBuffer, vertices.data(), sizeof(vertices));
        indexBuffer = createBuffer(sizeof(indices), buildInputUsage, hostMemory, true);
        uploadBuffer(indexBuffer, indices.data(), sizeof(indices));
    }

    AccelerationStructureResource createAccelerationStructure(vk::AccelerationStructureTypeKHR type,
                                                              vk::DeviceSize size) {
        AccelerationStructureResource resource;
        resource.storage = createBuffer(size,
                                        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                                            vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                        vk::MemoryPropertyFlagBits::eDeviceLocal, true);
        vk::AccelerationStructureCreateInfoKHR createInfo{
            .buffer = *resource.storage.buffer,
            .offset = 0,
            .size = size,
            .type = type,
        };
        resource.handle = vk::raii::AccelerationStructureKHR(device, createInfo);
        return resource;
    }

    BufferResource createScratchBuffer(vk::DeviceSize requiredSize) {
        const auto alignment =
            static_cast<vk::DeviceSize>(accelerationStructureProperties.minAccelerationStructureScratchOffsetAlignment);
        return createBuffer(requiredSize + alignment - 1,
                            vk::BufferUsageFlagBits::eStorageBuffer |
                                vk::BufferUsageFlagBits::eShaderDeviceAddress,
                            vk::MemoryPropertyFlagBits::eDeviceLocal, true);
    }

    vk::DeviceAddress alignedScratchAddress(const BufferResource &scratch) const {
        const auto alignment =
            static_cast<vk::DeviceAddress>(accelerationStructureProperties.minAccelerationStructureScratchOffsetAlignment);
        return alignUp(getBufferAddress(scratch), alignment);
    }

    void createBottomLevelAccelerationStructure() {
        vk::AccelerationStructureGeometryTrianglesDataKHR triangles{
            .vertexFormat = vk::Format::eR32G32B32Sfloat,
            .vertexData = {.deviceAddress = getBufferAddress(vertexBuffer)},
            .vertexStride = sizeof(Vertex),
            .maxVertex = 2,
            .indexType = vk::IndexType::eUint32,
            .indexData = {.deviceAddress = getBufferAddress(indexBuffer)},
        };
        vk::AccelerationStructureGeometryKHR geometry{
            .geometryType = vk::GeometryTypeKHR::eTriangles,
            .geometry = {.triangles = triangles},
            .flags = vk::GeometryFlagBitsKHR::eOpaque,
        };
        vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{
            .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
            .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
            .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
            .geometryCount = 1,
            .pGeometries = &geometry,
        };
        constexpr std::array<std::uint32_t, 1> primitiveCounts = {1};
        const auto sizes = device.getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo, primitiveCounts);

        bottomLevelAS =
            createAccelerationStructure(vk::AccelerationStructureTypeKHR::eBottomLevel, sizes.accelerationStructureSize);
        auto scratch = createScratchBuffer(sizes.buildScratchSize);

        buildInfo.dstAccelerationStructure = *bottomLevelAS.handle;
        buildInfo.scratchData.deviceAddress = alignedScratchAddress(scratch);
        vk::AccelerationStructureBuildRangeInfoKHR rangeInfo{
            .primitiveCount = 1,
            .primitiveOffset = 0,
            .firstVertex = 0,
            .transformOffset = 0,
        };
        const std::array buildInfos = {buildInfo};
        const std::array<const vk::AccelerationStructureBuildRangeInfoKHR *, 1> rangeInfos = {&rangeInfo};
        submitImmediate([&](vk::raii::CommandBuffer &commandBuffer) {
            commandBuffer.buildAccelerationStructuresKHR(buildInfos, rangeInfos);
        });
    }

    void createTopLevelAccelerationStructure() {
        const vk::AccelerationStructureDeviceAddressInfoKHR addressInfo{
            .accelerationStructure = *bottomLevelAS.handle,
        };
        const auto bottomLevelAddress = device.getAccelerationStructureAddressKHR(addressInfo);

        vk::AccelerationStructureInstanceKHR instanceData{};
        instanceData.transform.matrix[0][0] = 1.0f;
        instanceData.transform.matrix[1][1] = 1.0f;
        instanceData.transform.matrix[2][2] = 1.0f;
        instanceData.instanceCustomIndex = 0;
        instanceData.mask = 0xff;
        instanceData.instanceShaderBindingTableRecordOffset = 0;
        instanceData.flags =
            static_cast<std::uint32_t>(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);
        instanceData.accelerationStructureReference = bottomLevelAddress;

        constexpr auto hostMemory =
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
        instanceBuffer = createBuffer(sizeof(instanceData),
                                      vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
                                          vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                      hostMemory, true);
        uploadBuffer(instanceBuffer, &instanceData, sizeof(instanceData));

        vk::AccelerationStructureGeometryInstancesDataKHR instances{
            .arrayOfPointers = false,
            .data = {.deviceAddress = getBufferAddress(instanceBuffer)},
        };
        vk::AccelerationStructureGeometryKHR geometry{
            .geometryType = vk::GeometryTypeKHR::eInstances,
            .geometry = {.instances = instances},
        };
        vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{
            .type = vk::AccelerationStructureTypeKHR::eTopLevel,
            .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
            .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
            .geometryCount = 1,
            .pGeometries = &geometry,
        };
        constexpr std::array<std::uint32_t, 1> primitiveCounts = {1};
        const auto sizes = device.getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo, primitiveCounts);

        topLevelAS =
            createAccelerationStructure(vk::AccelerationStructureTypeKHR::eTopLevel, sizes.accelerationStructureSize);
        auto scratch = createScratchBuffer(sizes.buildScratchSize);

        buildInfo.dstAccelerationStructure = *topLevelAS.handle;
        buildInfo.scratchData.deviceAddress = alignedScratchAddress(scratch);
        vk::AccelerationStructureBuildRangeInfoKHR rangeInfo{
            .primitiveCount = 1,
            .primitiveOffset = 0,
            .firstVertex = 0,
            .transformOffset = 0,
        };
        const std::array buildInfos = {buildInfo};
        const std::array<const vk::AccelerationStructureBuildRangeInfoKHR *, 1> rangeInfos = {&rangeInfo};
        submitImmediate([&](vk::raii::CommandBuffer &commandBuffer) {
            commandBuffer.buildAccelerationStructuresKHR(buildInfos, rangeInfos);
        });
    }

    void createAccelerationStructures() {
        createBottomLevelAccelerationStructure();
        createTopLevelAccelerationStructure();
    }

    void createDescriptors() {
        const std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {
            vk::DescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = vk::DescriptorType::eAccelerationStructureKHR,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eRaygenKHR,
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 1,
                .descriptorType = vk::DescriptorType::eStorageImage,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eRaygenKHR,
            },
        };
        descriptorSetLayout = device.createDescriptorSetLayout({
            .bindingCount = static_cast<std::uint32_t>(bindings.size()),
            .pBindings = bindings.data(),
        });

        const std::array<vk::DescriptorPoolSize, 2> poolSizes = {
            vk::DescriptorPoolSize{.type = vk::DescriptorType::eAccelerationStructureKHR, .descriptorCount = 1},
            vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageImage, .descriptorCount = 1},
        };
        descriptorPool = device.createDescriptorPool({
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = 1,
            .poolSizeCount = static_cast<std::uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data(),
        });

        const vk::DescriptorSetLayout layout = *descriptorSetLayout;
        descriptorSets = device.allocateDescriptorSets({
            .descriptorPool = *descriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &layout,
        });

        const vk::AccelerationStructureKHR topLevelHandle = *topLevelAS.handle;
        vk::WriteDescriptorSetAccelerationStructureKHR accelerationStructureWrite{
            .accelerationStructureCount = 1,
            .pAccelerationStructures = &topLevelHandle,
        };
        vk::DescriptorImageInfo imageInfo{
            .imageView = *outputImage.view,
            .imageLayout = vk::ImageLayout::eGeneral,
        };
        const std::array<vk::WriteDescriptorSet, 2> writes = {
            vk::WriteDescriptorSet{
                .pNext = &accelerationStructureWrite,
                .dstSet = *descriptorSets.front(),
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eAccelerationStructureKHR,
            },
            vk::WriteDescriptorSet{
                .dstSet = *descriptorSets.front(),
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageImage,
                .pImageInfo = &imageInfo,
            },
        };
        device.updateDescriptorSets(writes, {});
    }

    void createRayTracingPipeline() {
        const std::filesystem::path shaderDirectory{SHADER_SPV_DIR};
        const std::array shaderPaths = {
            shaderDirectory / "raygen.rgen.spv",
            shaderDirectory / "miss.rmiss.spv",
            shaderDirectory / "closesthit.rchit.spv",
        };

        shaderModules.reserve(shaderPaths.size());
        for (const auto &path : shaderPaths) {
            const auto code = readSpirvFile(path);
            shaderModules.emplace_back(device, vk::ShaderModuleCreateInfo{
                                                   .codeSize = code.size() * sizeof(std::uint32_t),
                                                   .pCode = code.data(),
                                               });
        }

        const std::array<vk::PipelineShaderStageCreateInfo, 3> stages = {
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eRaygenKHR,
                .module = *shaderModules[0],
                .pName = "main",
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eMissKHR,
                .module = *shaderModules[1],
                .pName = "main",
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eClosestHitKHR,
                .module = *shaderModules[2],
                .pName = "main",
            },
        };
        const std::array<vk::RayTracingShaderGroupCreateInfoKHR, 3> groups = {
            vk::RayTracingShaderGroupCreateInfoKHR{
                .type = vk::RayTracingShaderGroupTypeKHR::eGeneral,
                .generalShader = 0,
                .closestHitShader = vk::ShaderUnusedKHR,
                .anyHitShader = vk::ShaderUnusedKHR,
                .intersectionShader = vk::ShaderUnusedKHR,
            },
            vk::RayTracingShaderGroupCreateInfoKHR{
                .type = vk::RayTracingShaderGroupTypeKHR::eGeneral,
                .generalShader = 1,
                .closestHitShader = vk::ShaderUnusedKHR,
                .anyHitShader = vk::ShaderUnusedKHR,
                .intersectionShader = vk::ShaderUnusedKHR,
            },
            vk::RayTracingShaderGroupCreateInfoKHR{
                .type = vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                .generalShader = vk::ShaderUnusedKHR,
                .closestHitShader = 2,
                .anyHitShader = vk::ShaderUnusedKHR,
                .intersectionShader = vk::ShaderUnusedKHR,
            },
        };

        const vk::DescriptorSetLayout descriptorLayout = *descriptorSetLayout;
        pipelineLayout = device.createPipelineLayout({
            .setLayoutCount = 1,
            .pSetLayouts = &descriptorLayout,
        });

        vk::RayTracingPipelineCreateInfoKHR createInfo{
            .stageCount = static_cast<std::uint32_t>(stages.size()),
            .pStages = stages.data(),
            .groupCount = static_cast<std::uint32_t>(groups.size()),
            .pGroups = groups.data(),
            .maxPipelineRayRecursionDepth = 1,
            .layout = *pipelineLayout,
        };
        rayTracingPipeline = vk::raii::Pipeline(device, nullptr, nullptr, createInfo);
    }

    void createShaderBindingTable() {
        constexpr std::uint32_t groupCount = 3;
        const auto handleSize = rayTracingProperties.shaderGroupHandleSize;
        const auto handleAlignment = rayTracingProperties.shaderGroupHandleAlignment;
        const auto baseAlignment = rayTracingProperties.shaderGroupBaseAlignment;
        const auto handleStride = alignUp(handleSize, handleAlignment);

        const vk::DeviceSize raygenOffset = 0;
        const vk::DeviceSize missOffset = alignUp<vk::DeviceSize>(handleStride, baseAlignment);
        const vk::DeviceSize hitOffset = alignUp<vk::DeviceSize>(missOffset + handleStride, baseAlignment);
        const vk::DeviceSize tableSize = hitOffset + handleStride;

        constexpr auto hostMemory =
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
        shaderBindingTable = createBuffer(tableSize + baseAlignment - 1,
                                          vk::BufferUsageFlagBits::eShaderBindingTableKHR |
                                              vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                          hostMemory, true);

        const auto bufferAddress = getBufferAddress(shaderBindingTable);
        const auto tableAddress = alignUp(bufferAddress, static_cast<vk::DeviceAddress>(baseAlignment));
        const auto hostOffset = static_cast<vk::DeviceSize>(tableAddress - bufferAddress);
        const auto handles = rayTracingPipeline.getRayTracingShaderGroupHandlesKHR<std::uint8_t>(
            0, groupCount, static_cast<std::size_t>(groupCount) * handleSize);

        auto *mapped = static_cast<std::uint8_t *>(
            shaderBindingTable.memory.mapMemory(0, shaderBindingTable.size));
        std::memset(mapped, 0, static_cast<std::size_t>(shaderBindingTable.size));
        std::memcpy(mapped + hostOffset + raygenOffset, handles.data(), handleSize);
        std::memcpy(mapped + hostOffset + missOffset, handles.data() + handleSize, handleSize);
        std::memcpy(mapped + hostOffset + hitOffset, handles.data() + 2 * handleSize, handleSize);
        shaderBindingTable.memory.unmapMemory();

        raygenRegion = {
            .deviceAddress = tableAddress + raygenOffset,
            .stride = handleStride,
            .size = handleStride,
        };
        missRegion = {
            .deviceAddress = tableAddress + missOffset,
            .stride = handleStride,
            .size = handleStride,
        };
        hitRegion = {
            .deviceAddress = tableAddress + hitOffset,
            .stride = handleStride,
            .size = handleStride,
        };
        callableRegion = {};
    }

    void createCommandBuffers() {
        vk::CommandBufferAllocateInfo allocateInfo{
            .commandPool = *commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = maxConcurrentFrames,
        };
        commandBuffers = device.allocateCommandBuffers(allocateInfo);
    }

    void recordCommandBuffer(std::uint32_t frameIndex, std::uint32_t imageIndex) {
        auto &commandBuffer = commandBuffers[frameIndex];
        commandBuffer.reset({});
        commandBuffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *rayTracingPipeline);
        const std::array descriptorSetHandles = {*descriptorSets.front()};
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, *pipelineLayout, 0,
                                         descriptorSetHandles, {});
        commandBuffer.traceRaysKHR(raygenRegion, missRegion, hitRegion, callableRegion, swapchainExtent.width,
                                   swapchainExtent.height, 1);

        vk::ImageMemoryBarrier outputToTransfer{
            .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
            .dstAccessMask = vk::AccessFlagBits::eTransferRead,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = *outputImage.image,
            .subresourceRange = colorSubresourceRange(),
        };
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eRayTracingShaderKHR,
                                      vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, {outputToTransfer});

        vk::ImageMemoryBarrier swapchainToTransfer{
            .srcAccessMask = {},
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = swapchainImages[imageIndex],
            .subresourceRange = colorSubresourceRange(),
        };
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                                      vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, {swapchainToTransfer});

        vk::ImageBlit blit{};
        blit.srcSubresource = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };
        blit.srcOffsets[0] = {.x = 0, .y = 0, .z = 0};
        blit.srcOffsets[1] = {
            .x = static_cast<std::int32_t>(swapchainExtent.width),
            .y = static_cast<std::int32_t>(swapchainExtent.height),
            .z = 1,
        };
        blit.dstSubresource = blit.srcSubresource;
        blit.dstOffsets[0] = {.x = 0, .y = 0, .z = 0};
        blit.dstOffsets[1] = blit.srcOffsets[1];
        commandBuffer.blitImage(*outputImage.image, vk::ImageLayout::eTransferSrcOptimal, swapchainImages[imageIndex],
                                vk::ImageLayout::eTransferDstOptimal, {blit}, vk::Filter::eNearest);

        vk::ImageMemoryBarrier outputToGeneral{
            .srcAccessMask = vk::AccessFlagBits::eTransferRead,
            .dstAccessMask = vk::AccessFlagBits::eShaderWrite,
            .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
            .newLayout = vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = *outputImage.image,
            .subresourceRange = colorSubresourceRange(),
        };
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                      vk::PipelineStageFlagBits::eRayTracingShaderKHR, {}, {}, {}, {outputToGeneral});

        vk::ImageMemoryBarrier swapchainToPresent{
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = {},
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::ePresentSrcKHR,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = swapchainImages[imageIndex],
            .subresourceRange = colorSubresourceRange(),
        };
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                      vk::PipelineStageFlagBits::eBottomOfPipe, {}, {}, {}, {swapchainToPresent});
        commandBuffer.end();
    }

    void createSyncObjects() {
        imageAvailableSemaphores.reserve(maxConcurrentFrames);
        inFlightFences.reserve(maxConcurrentFrames);
        renderFinishedSemaphores.reserve(swapchainImages.size());

        for (std::uint32_t index = 0; index < maxConcurrentFrames; ++index) {
            imageAvailableSemaphores.push_back(device.createSemaphore({}));
            inFlightFences.push_back(device.createFence({.flags = vk::FenceCreateFlagBits::eSignaled}));
        }
        for (std::size_t index = 0; index < swapchainImages.size(); ++index) {
            renderFinishedSemaphores.push_back(device.createSemaphore({}));
        }
    }

    void drawFrame() {
        const auto frameIndex = currentFrame;
        (void)device.waitForFences({*inFlightFences[frameIndex]}, true,
                                   std::numeric_limits<std::uint64_t>::max());

        std::uint32_t imageIndex = 0;
        try {
            const auto [result, acquiredIndex] = swapchain.acquireNextImage(
                std::numeric_limits<std::uint64_t>::max(), *imageAvailableSemaphores[frameIndex], vk::Fence{});
            (void)result;
            imageIndex = acquiredIndex;
        } catch (const vk::SystemError &error) {
            if (error.code().value() == static_cast<int>(vk::Result::eErrorOutOfDateKHR)) {
                throw std::runtime_error("Swapchain became out of date; this fixed-size example does not recreate it");
            }
            throw;
        }

        device.resetFences({*inFlightFences[frameIndex]});
        recordCommandBuffer(frameIndex, imageIndex);

        const vk::Semaphore waitSemaphore = *imageAvailableSemaphores[frameIndex];
        const vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eTransfer;
        const vk::CommandBuffer commandBuffer = *commandBuffers[frameIndex];
        const vk::Semaphore signalSemaphore = *renderFinishedSemaphores[imageIndex];
        vk::SubmitInfo submitInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &waitSemaphore,
            .pWaitDstStageMask = &waitStage,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &signalSemaphore,
        };
        graphicsQueue.submit(submitInfo, *inFlightFences[frameIndex]);

        const vk::SwapchainKHR swapchainHandle = *swapchain;
        vk::PresentInfoKHR presentInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &signalSemaphore,
            .swapchainCount = 1,
            .pSwapchains = &swapchainHandle,
            .pImageIndices = &imageIndex,
        };
        try {
            (void)graphicsQueue.presentKHR(presentInfo);
        } catch (const vk::SystemError &error) {
            if (error.code().value() == static_cast<int>(vk::Result::eErrorOutOfDateKHR)) {
                throw std::runtime_error("Swapchain became out of date; this fixed-size example does not recreate it");
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
        RayTracingApplication application;
        application.run();
    } catch (const std::exception &error) {
        std::println("Error: {}", error.what());
        return 1;
    }
    return 0;
}
