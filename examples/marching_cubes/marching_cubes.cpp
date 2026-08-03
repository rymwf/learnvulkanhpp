#include <glm/ext/matrix_clip_space.hpp>
#include <glm/glm.hpp>
#include <imgui.h>

import vulkan;
import std;
import common;
import common.imgui_overlay;

#ifndef SHADER_SPV_DIR
#error "SHADER_SPV_DIR must be set by CMake (see examples/marching_cubes/CMakeLists.txt)"
#endif

namespace {

constexpr std::uint32_t minimumResolution = 12;
constexpr std::uint32_t maximumResolution = 40;
constexpr std::uint32_t defaultResolution = 32;
constexpr std::uint32_t maximumVerticesPerCell = 15;
constexpr std::uint32_t generatedVertexStride = 8 * sizeof(float);
constexpr std::uint32_t maximumGeneratedVertices =
    maximumResolution * maximumResolution * maximumResolution * maximumVerticesPerCell;
constexpr vk::DeviceSize generatedVertexBufferSize =
    static_cast<vk::DeviceSize>(maximumGeneratedVertices) * generatedVertexStride;
constexpr vk::Format volumeFormat = vk::Format::eR32Sfloat;
constexpr std::uint32_t queriesPerFrame = 3;
constexpr float cameraVerticalFovRadians = 0.78539816339744830962f;
constexpr float orbitRadiansPerPixel = 0.005f;

struct BufferResource {
    vk::raii::DeviceMemory memory = nullptr;
    vk::raii::Buffer buffer = nullptr;
};

struct ImageResource {
    vk::raii::DeviceMemory memory = nullptr;
    vk::raii::Image image = nullptr;
    vk::raii::ImageView imageView = nullptr;
};

struct alignas(16) ComputePushConstants {
    float time = 0.0f;
    float isoLevel = 1.0f;
    std::uint32_t resolution = defaultResolution;
    float padding = 0.0f;
};

struct alignas(16) SurfacePushConstants {
    glm::mat4 viewProjection{1.0f};
    glm::vec4 colorAndWireframe{0.10f, 0.58f, 0.74f, 0.0f};
};

struct alignas(16) SlicePushConstants {
    float isoLevel = 1.0f;
    std::uint32_t resolution = defaultResolution;
    std::int32_t axis = 2;
    std::int32_t sliceIndex = defaultResolution / 2;
};

static_assert(sizeof(ComputePushConstants) == 16);
static_assert(sizeof(SurfacePushConstants) == 80);
static_assert(sizeof(SlicePushConstants) == 16);
static_assert(sizeof(vk::DrawIndirectCommand) == 4 * sizeof(std::uint32_t));

struct Parameters {
    int resolution = static_cast<int>(defaultResolution);
    float isoLevel = 1.0f;
    float animationSpeed = 1.0f;
    bool paused = false;
    bool wireframe = false;
    bool showSlice = true;
    int sliceAxis = 2;
    int sliceIndex = static_cast<int>(defaultResolution / 2);
};

struct FrameResultSlot {
    BufferResource vertexCountReadback;
    bool pending = false;
    std::uint32_t resolution = defaultResolution;
};

struct GenerationStatistics {
    bool valid = false;
    bool timingsValid = false;
    std::uint32_t resolution = defaultResolution;
    std::uint32_t vertexCount = 0;
    double volumeMilliseconds = 0.0;
    double extractionMilliseconds = 0.0;
};

[[nodiscard]] std::uint32_t dispatchGroupCount(std::uint32_t invocationCount) {
    return (invocationCount + 3u) / 4u;
}

}  // namespace

class MarchingCubesApplication final : public WindowedVulkanApp {
public:
    MarchingCubesApplication()
        : WindowedVulkanApp({
              .window = {
                  .width = 1120,
                  .height = 760,
                  .title = "Vulkan GPU Marching Cubes",
              },
              .apiVersion = vk::ApiVersion11,
              .requiredQueueFlags = vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute,
          }),
          camera(OrbitCameraConfig{
              .target = {0.0f, 0.0f, 0.0f},
              .yawRadians = 2.35f,
              .pitchRadians = 0.34f,
              .distance = 4.8f,
              .minDistance = 2.4f,
              .maxDistance = 10.0f,
          }) {}

private:
    ImageResource volumeImage;
    BufferResource generatedVertices;
    BufferResource indirectCommand;
    std::vector<FrameResultSlot> frameResultSlots;

    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> descriptorSets;

    vk::raii::ShaderModule fieldShader = nullptr;
    vk::raii::ShaderModule marchingCubesShader = nullptr;
    vk::raii::ShaderModule surfaceVertexShader = nullptr;
    vk::raii::ShaderModule surfaceFragmentShader = nullptr;
    vk::raii::ShaderModule sliceVertexShader = nullptr;
    vk::raii::ShaderModule sliceFragmentShader = nullptr;

    vk::raii::PipelineLayout computePipelineLayout = nullptr;
    vk::raii::PipelineLayout surfacePipelineLayout = nullptr;
    vk::raii::PipelineLayout slicePipelineLayout = nullptr;
    vk::raii::Pipeline fieldPipeline = nullptr;
    vk::raii::Pipeline marchingCubesPipeline = nullptr;
    vk::raii::Pipeline surfacePipeline = nullptr;
    vk::raii::Pipeline slicePipeline = nullptr;

    vk::raii::QueryPool timestampQueryPool = nullptr;
    std::uint32_t timestampValidBits = 0;
    double timestampPeriodNanoseconds = 0.0;

    OrbitCamera camera;
    std::optional<GlfwCursorPosition> previousDragCursor;
    bool resetKeyWasPressed = false;
    Parameters parameters;
    GenerationStatistics statistics;
    float simulationTime = 0.0f;
    std::chrono::steady_clock::time_point previousFrameTime = std::chrono::steady_clock::now();
    vk::ImageLayout volumeImageLayout = vk::ImageLayout::eUndefined;
    bool fieldDirty = true;
    bool meshDirty = true;
    bool meshHasBeenGenerated = false;
    ImGuiVulkanOverlay overlay;

    bool supportsAdditionalDeviceRequirements(const vk::raii::PhysicalDevice &candidate) const override {
        const auto formatProperties = candidate.getFormatProperties(volumeFormat);
        if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eStorageImage)) {
            return false;
        }

        const auto limits = candidate.getProperties().limits;
        return limits.maxImageDimension3D >= maximumResolution + 1 &&
               static_cast<vk::DeviceSize>(limits.maxStorageBufferRange) >= generatedVertexBufferSize;
    }

    void createApplicationResources() override {
        createVolumeImage();
        createMeshBuffers();
        createDescriptors();
        createPipelineLayouts();
        createShaderModules();
        createComputePipelines();
        createGraphicsPipelines();
        createStatisticsResources();
        overlay.initialize(makeOverlayConfig());
    }

    void releaseSwapchainDependentResources() override {
        overlay.releaseRenderer();
        slicePipeline = nullptr;
        surfacePipeline = nullptr;
    }

    void recreateSwapchainDependentResources() override {
        createGraphicsPipelines();
        overlay.recreateRenderer(swapchainRenderPass(), imguiMinImageCount(),
                                 static_cast<std::uint32_t>(swapchainImages().size()));
    }

    void recordFrameCommands(vk::raii::CommandBuffer &commandBuffer, std::uint32_t imageIndex) override {
        const auto frameIndex = currentFrameIndex();
        readCompletedStatistics(frameIndex);

        overlay.beginFrame();
        drawControlPanel();
        updateInput();
        updateSimulationTime();

        recordGenerationCommands(commandBuffer, frameIndex);
        recordRenderingCommands(commandBuffer, imageIndex);
    }

    void createVolumeImage() {
        constexpr std::uint32_t sampleCount = maximumResolution + 1;
        volumeImage.image = device().createImage({
            .imageType = vk::ImageType::e3D,
            .format = volumeFormat,
            .extent = {sampleCount, sampleCount, sampleCount},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eStorage,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined,
        });

        const auto requirements = volumeImage.image.getMemoryRequirements();
        volumeImage.memory = vk::raii::DeviceMemory(
            device(), {
                          .allocationSize = requirements.size,
                          .memoryTypeIndex = findMemoryTypeIndex(requirements, vk::MemoryPropertyFlagBits::eDeviceLocal),
                      });
        volumeImage.image.bindMemory(*volumeImage.memory, 0);
        volumeImage.imageView = device().createImageView({
            .image = *volumeImage.image,
            .viewType = vk::ImageViewType::e3D,
            .format = volumeFormat,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        });
    }

    [[nodiscard]] BufferResource createBuffer(vk::DeviceSize size,
                                              vk::BufferUsageFlags usage,
                                              vk::MemoryPropertyFlags memoryProperties) {
        BufferResource resource;
        resource.buffer = device().createBuffer({
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
        });
        const auto requirements = resource.buffer.getMemoryRequirements();
        resource.memory = vk::raii::DeviceMemory(
            device(), {
                          .allocationSize = requirements.size,
                          .memoryTypeIndex = findMemoryTypeIndex(requirements, memoryProperties),
                      });
        resource.buffer.bindMemory(*resource.memory, 0);
        return resource;
    }

    void createMeshBuffers() {
        generatedVertices = createBuffer(generatedVertexBufferSize, vk::BufferUsageFlagBits::eStorageBuffer,
                                         vk::MemoryPropertyFlagBits::eDeviceLocal);
        indirectCommand = createBuffer(sizeof(vk::DrawIndirectCommand),
                                       vk::BufferUsageFlagBits::eStorageBuffer |
                                           vk::BufferUsageFlagBits::eIndirectBuffer |
                                           vk::BufferUsageFlagBits::eTransferDst |
                                           vk::BufferUsageFlagBits::eTransferSrc,
                                       vk::MemoryPropertyFlagBits::eDeviceLocal);
    }

    void createDescriptors() {
        const std::array bindings = {
            vk::DescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = vk::DescriptorType::eStorageImage,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eFragment,
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eVertex,
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 2,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eCompute,
            },
        };
        descriptorSetLayout = device().createDescriptorSetLayout({
            .bindingCount = static_cast<std::uint32_t>(bindings.size()),
            .pBindings = bindings.data(),
        });

        const std::array poolSizes = {
            vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageImage, .descriptorCount = 1},
            vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 2},
        };
        descriptorPool = device().createDescriptorPool({
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = 1,
            .poolSizeCount = static_cast<std::uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data(),
        });

        const vk::DescriptorSetLayout rawLayout = *descriptorSetLayout;
        descriptorSets = device().allocateDescriptorSets({
            .descriptorPool = *descriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &rawLayout,
        });

        const vk::DescriptorImageInfo volumeInfo{
            .imageView = *volumeImage.imageView,
            .imageLayout = vk::ImageLayout::eGeneral,
        };
        const vk::DescriptorBufferInfo vertexInfo{
            .buffer = *generatedVertices.buffer,
            .offset = 0,
            .range = generatedVertexBufferSize,
        };
        const vk::DescriptorBufferInfo indirectInfo{
            .buffer = *indirectCommand.buffer,
            .offset = 0,
            .range = sizeof(vk::DrawIndirectCommand),
        };
        const std::array writes = {
            vk::WriteDescriptorSet{
                .dstSet = *descriptorSets.front(),
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageImage,
                .pImageInfo = &volumeInfo,
            },
            vk::WriteDescriptorSet{
                .dstSet = *descriptorSets.front(),
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &vertexInfo,
            },
            vk::WriteDescriptorSet{
                .dstSet = *descriptorSets.front(),
                .dstBinding = 2,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &indirectInfo,
            },
        };
        device().updateDescriptorSets(writes, {});
    }

    void createPipelineLayouts() {
        const vk::DescriptorSetLayout rawLayout = *descriptorSetLayout;
        const vk::PushConstantRange computeRange{
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset = 0,
            .size = sizeof(ComputePushConstants),
        };
        computePipelineLayout = device().createPipelineLayout({
            .setLayoutCount = 1,
            .pSetLayouts = &rawLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &computeRange,
        });

        const vk::PushConstantRange surfaceRange{
            .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            .offset = 0,
            .size = sizeof(SurfacePushConstants),
        };
        surfacePipelineLayout = device().createPipelineLayout({
            .setLayoutCount = 1,
            .pSetLayouts = &rawLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &surfaceRange,
        });

        const vk::PushConstantRange sliceRange{
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
            .offset = 0,
            .size = sizeof(SlicePushConstants),
        };
        slicePipelineLayout = device().createPipelineLayout({
            .setLayoutCount = 1,
            .pSetLayouts = &rawLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &sliceRange,
        });
    }

    [[nodiscard]] vk::raii::ShaderModule loadShaderModule(const std::filesystem::path &path) {
        const auto code = readSpirvFile(path);
        return device().createShaderModule({
            .codeSize = code.size() * sizeof(std::uint32_t),
            .pCode = code.data(),
        });
    }

    void createShaderModules() {
        const std::filesystem::path shaderDirectory{SHADER_SPV_DIR};
        fieldShader = loadShaderModule(shaderDirectory / "field.comp.spv");
        marchingCubesShader = loadShaderModule(shaderDirectory / "marching_cubes.comp.spv");
        surfaceVertexShader = loadShaderModule(shaderDirectory / "surface.vert.spv");
        surfaceFragmentShader = loadShaderModule(shaderDirectory / "surface.frag.spv");
        sliceVertexShader = loadShaderModule(shaderDirectory / "slice.vert.spv");
        sliceFragmentShader = loadShaderModule(shaderDirectory / "slice.frag.spv");
    }

    void createComputePipelines() {
        fieldPipeline = device().createComputePipeline(nullptr, {
                                                                    .stage = {
                                                                        .stage = vk::ShaderStageFlagBits::eCompute,
                                                                        .module = *fieldShader,
                                                                        .pName = "main",
                                                                    },
                                                                    .layout = *computePipelineLayout,
                                                                });
        marchingCubesPipeline = device().createComputePipeline(nullptr, {
                                                                           .stage = {
                                                                               .stage = vk::ShaderStageFlagBits::eCompute,
                                                                               .module = *marchingCubesShader,
                                                                               .pName = "main",
                                                                           },
                                                                           .layout = *computePipelineLayout,
                                                                       });
    }

    void createGraphicsPipelines() {
        createSurfacePipeline();
        createSlicePipeline();
    }

    void createSurfacePipeline() {
        const std::array stages = {
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = *surfaceVertexShader,
                .pName = "main",
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = *surfaceFragmentShader,
                .pName = "main",
            },
        };
        const vk::PipelineVertexInputStateCreateInfo vertexInput{};
        const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::eTriangleList,
        };
        const auto extent = swapchainExtent();
        const vk::Viewport viewport{
            .x = 0.0f,
            .y = static_cast<float>(extent.height),
            .width = static_cast<float>(extent.width),
            .height = -static_cast<float>(extent.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        const vk::Rect2D scissor{.offset = {0, 0}, .extent = extent};
        const vk::PipelineViewportStateCreateInfo viewportState{
            .viewportCount = 1,
            .pViewports = &viewport,
            .scissorCount = 1,
            .pScissors = &scissor,
        };
        const vk::PipelineRasterizationStateCreateInfo rasterizer{
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eNone,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .lineWidth = 1.0f,
        };
        const vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
        };
        const vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = true,
            .depthWriteEnable = true,
            .depthCompareOp = vk::CompareOp::eLess,
        };
        const vk::PipelineColorBlendAttachmentState blendAttachment{
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
        };
        const vk::PipelineColorBlendStateCreateInfo blending{
            .attachmentCount = 1,
            .pAttachments = &blendAttachment,
        };
        surfacePipeline = device().createGraphicsPipeline(nullptr, {
                                                                       .stageCount = stages.size(),
                                                                       .pStages = stages.data(),
                                                                       .pVertexInputState = &vertexInput,
                                                                       .pInputAssemblyState = &inputAssembly,
                                                                       .pViewportState = &viewportState,
                                                                       .pRasterizationState = &rasterizer,
                                                                       .pMultisampleState = &multisampling,
                                                                       .pDepthStencilState = &depthStencil,
                                                                       .pColorBlendState = &blending,
                                                                       .layout = *surfacePipelineLayout,
                                                                       .renderPass = swapchainRenderPass(),
                                                                       .subpass = 0,
                                                                   });
    }

    void createSlicePipeline() {
        const std::array stages = {
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = *sliceVertexShader,
                .pName = "main",
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = *sliceFragmentShader,
                .pName = "main",
            },
        };
        const vk::PipelineVertexInputStateCreateInfo vertexInput{};
        const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::eTriangleList,
        };
        const vk::PipelineViewportStateCreateInfo viewportState{
            .viewportCount = 1,
            .scissorCount = 1,
        };
        const vk::PipelineRasterizationStateCreateInfo rasterizer{
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eNone,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .lineWidth = 1.0f,
        };
        const vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
        };
        const vk::PipelineDepthStencilStateCreateInfo depthStencil{};
        const vk::PipelineColorBlendAttachmentState blendAttachment{
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
        };
        const vk::PipelineColorBlendStateCreateInfo blending{
            .attachmentCount = 1,
            .pAttachments = &blendAttachment,
        };
        const std::array dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
        const vk::PipelineDynamicStateCreateInfo dynamicState{
            .dynamicStateCount = dynamicStates.size(),
            .pDynamicStates = dynamicStates.data(),
        };
        slicePipeline = device().createGraphicsPipeline(nullptr, {
                                                                     .stageCount = stages.size(),
                                                                     .pStages = stages.data(),
                                                                     .pVertexInputState = &vertexInput,
                                                                     .pInputAssemblyState = &inputAssembly,
                                                                     .pViewportState = &viewportState,
                                                                     .pRasterizationState = &rasterizer,
                                                                     .pMultisampleState = &multisampling,
                                                                     .pDepthStencilState = &depthStencil,
                                                                     .pColorBlendState = &blending,
                                                                     .pDynamicState = &dynamicState,
                                                                     .layout = *slicePipelineLayout,
                                                                     .renderPass = swapchainRenderPass(),
                                                                     .subpass = 0,
                                                                 });
    }

    void createStatisticsResources() {
        frameResultSlots.reserve(framesInFlight());
        for (std::uint32_t index = 0; index < framesInFlight(); ++index) {
            FrameResultSlot slot;
            slot.vertexCountReadback = createBuffer(
                sizeof(std::uint32_t), vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
            frameResultSlots.push_back(std::move(slot));
        }

        const auto queueFamilies = physicalDevice().getQueueFamilyProperties();
        timestampValidBits = queueFamilies[graphicsQueueFamilyIndex()].timestampValidBits;
        if (timestampValidBits == 0) {
            return;
        }

        timestampPeriodNanoseconds = physicalDevice().getProperties().limits.timestampPeriod;
        timestampQueryPool = device().createQueryPool({
            .queryType = vk::QueryType::eTimestamp,
            .queryCount = framesInFlight() * queriesPerFrame,
        });
    }

    void readCompletedStatistics(std::uint32_t frameIndex) {
        auto &slot = frameResultSlots.at(frameIndex);
        if (!slot.pending) {
            return;
        }

        std::uint32_t vertexCount = 0;
        void *mapped = slot.vertexCountReadback.memory.mapMemory(0, sizeof(vertexCount));
        std::memcpy(&vertexCount, mapped, sizeof(vertexCount));
        slot.vertexCountReadback.memory.unmapMemory();

        statistics.valid = true;
        statistics.resolution = slot.resolution;
        statistics.vertexCount = std::min(vertexCount, maximumGeneratedVertices);
        statistics.timingsValid = false;

        if (timestampValidBits != 0) {
            std::array<std::uint64_t, queriesPerFrame> timestamps{};
            const auto firstQuery = frameIndex * queriesPerFrame;
            const auto result = timestampQueryPool.getResults(
                firstQuery, queriesPerFrame, sizeof(timestamps), timestamps.data(), sizeof(std::uint64_t),
                vk::QueryResultFlagBits::e64);
            if (result == vk::Result::eSuccess) {
                statistics.volumeMilliseconds = timestampDelta(timestamps[0], timestamps[1]);
                statistics.extractionMilliseconds = timestampDelta(timestamps[1], timestamps[2]);
                statistics.timingsValid = true;
            }
        }
        slot.pending = false;
    }

    [[nodiscard]] double timestampDelta(std::uint64_t begin, std::uint64_t end) const {
        const auto mask = timestampValidBits >= 64
                              ? std::numeric_limits<std::uint64_t>::max()
                              : (std::uint64_t{1} << timestampValidBits) - 1;
        const auto ticks = (end - begin) & mask;
        return static_cast<double>(ticks) * timestampPeriodNanoseconds * 1e-6;
    }

    void recordGenerationCommands(vk::raii::CommandBuffer &commandBuffer, std::uint32_t frameIndex) {
        if (!meshDirty) {
            return;
        }

        const auto queryBase = frameIndex * queriesPerFrame;
        if (timestampValidBits != 0) {
            commandBuffer.resetQueryPool(*timestampQueryPool, queryBase, queriesPerFrame);
            commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *timestampQueryPool, queryBase);
        }

        const ComputePushConstants pushConstants{
            .time = simulationTime,
            .isoLevel = parameters.isoLevel,
            .resolution = static_cast<std::uint32_t>(parameters.resolution),
        };
        const std::array descriptorSetHandles = {*descriptorSets.front()};

        if (fieldDirty) {
            transitionVolumeForWrite(commandBuffer);
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *fieldPipeline);
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computePipelineLayout, 0,
                                             descriptorSetHandles, {});
            commandBuffer.pushConstants(*computePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                                        sizeof(pushConstants), &pushConstants);
            const auto sampleCount = static_cast<std::uint32_t>(parameters.resolution + 1);
            const auto groups = dispatchGroupCount(sampleCount);
            commandBuffer.dispatch(groups, groups, groups);
            transitionVolumeForRead(commandBuffer);
        }

        if (timestampValidBits != 0) {
            commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, *timestampQueryPool,
                                         queryBase + 1);
        }

        prepareMeshForWrite(commandBuffer);
        const vk::DrawIndirectCommand resetCommand{
            .vertexCount = 0,
            .instanceCount = 1,
            .firstVertex = 0,
            .firstInstance = 0,
        };
        commandBuffer.updateBuffer(*indirectCommand.buffer, 0, sizeof(resetCommand), &resetCommand);
        transitionIndirectTransferToCompute(commandBuffer);

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *marchingCubesPipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computePipelineLayout, 0,
                                         descriptorSetHandles, {});
        commandBuffer.pushConstants(*computePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                                    sizeof(pushConstants), &pushConstants);
        const auto groups = dispatchGroupCount(static_cast<std::uint32_t>(parameters.resolution));
        commandBuffer.dispatch(groups, groups, groups);

        if (timestampValidBits != 0) {
            commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, *timestampQueryPool,
                                         queryBase + 2);
        }
        transitionMeshForRead(commandBuffer);

        auto &slot = frameResultSlots.at(frameIndex);
        const vk::BufferCopy countCopy{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = sizeof(std::uint32_t),
        };
        commandBuffer.copyBuffer(*indirectCommand.buffer, *slot.vertexCountReadback.buffer, {countCopy});
        slot.pending = true;
        slot.resolution = static_cast<std::uint32_t>(parameters.resolution);

        fieldDirty = false;
        meshDirty = false;
        meshHasBeenGenerated = true;
    }

    void transitionVolumeForWrite(vk::raii::CommandBuffer &commandBuffer) {
        const auto firstUse = volumeImageLayout == vk::ImageLayout::eUndefined;
        const vk::ImageMemoryBarrier barrier{
            .srcAccessMask = firstUse ? vk::AccessFlags{} : vk::AccessFlagBits::eShaderRead,
            .dstAccessMask = vk::AccessFlagBits::eShaderWrite,
            .oldLayout = volumeImageLayout,
            .newLayout = vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = *volumeImage.image,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        const auto sourceStage = firstUse
                                     ? vk::PipelineStageFlags{vk::PipelineStageFlagBits::eTopOfPipe}
                                     : vk::PipelineStageFlagBits::eComputeShader |
                                           vk::PipelineStageFlagBits::eFragmentShader;
        commandBuffer.pipelineBarrier(sourceStage, vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, {barrier});
        volumeImageLayout = vk::ImageLayout::eGeneral;
    }

    void transitionVolumeForRead(vk::raii::CommandBuffer &commandBuffer) {
        const vk::ImageMemoryBarrier barrier{
            .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = *volumeImage.image,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                      vk::PipelineStageFlagBits::eComputeShader |
                                          vk::PipelineStageFlagBits::eFragmentShader,
                                      {}, {}, {}, {barrier});
    }

    void prepareMeshForWrite(vk::raii::CommandBuffer &commandBuffer) {
        const auto sourceStage = meshHasBeenGenerated
                                     ? vk::PipelineStageFlagBits::eVertexShader |
                                           vk::PipelineStageFlagBits::eDrawIndirect |
                                           vk::PipelineStageFlagBits::eTransfer
                                     : vk::PipelineStageFlags{vk::PipelineStageFlagBits::eTopOfPipe};
        const std::array barriers = {
            vk::BufferMemoryBarrier{
                .srcAccessMask = meshHasBeenGenerated ? vk::AccessFlags{vk::AccessFlagBits::eShaderRead}
                                                      : vk::AccessFlags{},
                .dstAccessMask = vk::AccessFlagBits::eShaderWrite,
                .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                .buffer = *generatedVertices.buffer,
                .offset = 0,
                .size = generatedVertexBufferSize,
            },
            vk::BufferMemoryBarrier{
                .srcAccessMask = meshHasBeenGenerated
                                     ? vk::AccessFlagBits::eIndirectCommandRead | vk::AccessFlagBits::eTransferRead
                                     : vk::AccessFlags{},
                .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                .buffer = *indirectCommand.buffer,
                .offset = 0,
                .size = sizeof(vk::DrawIndirectCommand),
            },
        };
        commandBuffer.pipelineBarrier(sourceStage,
                                      vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
                                      {}, {}, barriers, {});
    }

    void transitionIndirectTransferToCompute(vk::raii::CommandBuffer &commandBuffer) {
        const vk::BufferMemoryBarrier barrier{
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .buffer = *indirectCommand.buffer,
            .offset = 0,
            .size = sizeof(vk::DrawIndirectCommand),
        };
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                      vk::PipelineStageFlagBits::eComputeShader, {}, {}, {barrier}, {});
    }

    void transitionMeshForRead(vk::raii::CommandBuffer &commandBuffer) {
        const std::array barriers = {
            vk::BufferMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                .dstAccessMask = vk::AccessFlagBits::eShaderRead,
                .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                .buffer = *generatedVertices.buffer,
                .offset = 0,
                .size = generatedVertexBufferSize,
            },
            vk::BufferMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                .dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead | vk::AccessFlagBits::eTransferRead,
                .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                .buffer = *indirectCommand.buffer,
                .offset = 0,
                .size = sizeof(vk::DrawIndirectCommand),
            },
        };
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                      vk::PipelineStageFlagBits::eVertexShader |
                                          vk::PipelineStageFlagBits::eDrawIndirect |
                                          vk::PipelineStageFlagBits::eTransfer,
                                      {}, {}, barriers, {});
    }

    void recordRenderingCommands(vk::raii::CommandBuffer &commandBuffer, std::uint32_t imageIndex) {
        const auto extent = swapchainExtent();
        const auto projection = glm::perspectiveRH_ZO(
            cameraVerticalFovRadians, static_cast<float>(extent.width) / static_cast<float>(std::max(extent.height, 1u)),
            0.1f, 20.0f);
        const SurfacePushConstants surfacePushConstants{
            .viewProjection = projection * camera.viewMatrix(),
            .colorAndWireframe = {0.10f, 0.58f, 0.74f, parameters.wireframe ? 1.0f : 0.0f},
        };
        const std::array descriptorSetHandles = {*descriptorSets.front()};

        std::array<vk::ClearValue, 2> clearValues{};
        clearValues[0].color.setFloat32(std::array<float, 4>{0.025f, 0.035f, 0.045f, 1.0f});
        clearValues[1].depthStencil = vk::ClearDepthStencilValue{1.0f, 0};
        beginSwapchainRenderPass(commandBuffer, imageIndex, clearValues);

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *surfacePipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *surfacePipelineLayout, 0,
                                         descriptorSetHandles, {});
        commandBuffer.pushConstants(*surfacePipelineLayout,
                                    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
                                    sizeof(surfacePushConstants), &surfacePushConstants);
        commandBuffer.drawIndirect(*indirectCommand.buffer, 0, 1, sizeof(vk::DrawIndirectCommand));

        if (shouldDrawSlicePreview(extent)) {
            drawSlicePreview(commandBuffer, extent, descriptorSetHandles);
        }
        overlay.render(commandBuffer);
        commandBuffer.endRenderPass();
    }

    [[nodiscard]] bool shouldDrawSlicePreview(vk::Extent2D extent) const {
        return parameters.showSlice && extent.width >= 480 && extent.height >= 280;
    }

    void drawSlicePreview(vk::raii::CommandBuffer &commandBuffer,
                          vk::Extent2D extent,
                          std::span<const vk::DescriptorSet> descriptorSetHandles) {
        const auto desiredSize = std::min(static_cast<float>(extent.width) * 0.25f,
                                          static_cast<float>(extent.height) * 0.34f);
        const auto previewSize = std::clamp(desiredSize, 96.0f, 240.0f);
        constexpr float margin = 12.0f;
        const auto previewX = static_cast<float>(extent.width) - previewSize - margin;
        const auto previewY = static_cast<float>(extent.height) - previewSize - margin;
        const vk::Viewport viewport{
            .x = previewX,
            .y = previewY + previewSize,
            .width = previewSize,
            .height = -previewSize,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        const vk::Rect2D scissor{
            .offset = {
                static_cast<std::int32_t>(previewX),
                static_cast<std::int32_t>(previewY),
            },
            .extent = {
                static_cast<std::uint32_t>(previewSize),
                static_cast<std::uint32_t>(previewSize),
            },
        };
        const SlicePushConstants pushConstants{
            .isoLevel = parameters.isoLevel,
            .resolution = static_cast<std::uint32_t>(parameters.resolution),
            .axis = parameters.sliceAxis,
            .sliceIndex = parameters.sliceIndex,
        };

        commandBuffer.setViewport(0, {viewport});
        commandBuffer.setScissor(0, {scissor});
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *slicePipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *slicePipelineLayout, 0,
                                         descriptorSetHandles, {});
        commandBuffer.pushConstants(*slicePipelineLayout, vk::ShaderStageFlagBits::eFragment, 0,
                                    sizeof(pushConstants), &pushConstants);
        commandBuffer.draw(3, 1, 0, 0);
    }

    void updateSimulationTime() {
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration<float>(now - previousFrameTime).count();
        previousFrameTime = now;
        if (parameters.paused || parameters.animationSpeed <= 0.0f) {
            return;
        }

        simulationTime += std::min(elapsed, 0.1f) * parameters.animationSpeed;
        fieldDirty = true;
        meshDirty = true;
    }

    void updateInput() {
        auto &inputWindow = window();
        const auto captureMouse = overlay.wantsMouseCapture();
        const auto cursor = inputWindow.cursorPosition();
        const auto panning = inputWindow.isPressed(GlfwMouseButton::Middle);
        const auto orbiting = inputWindow.isPressed(GlfwMouseButton::Left);

        if (!captureMouse && (panning || orbiting)) {
            if (previousDragCursor) {
                const auto horizontalPixels = static_cast<float>(cursor.x - previousDragCursor->x);
                const auto verticalPixels = static_cast<float>(cursor.y - previousDragCursor->y);
                if (panning) {
                    const auto extent = swapchainExtent();
                    camera.pan(horizontalPixels, verticalPixels, cameraVerticalFovRadians,
                               static_cast<float>(std::max(extent.height, 1u)));
                } else {
                    camera.orbit(horizontalPixels * orbitRadiansPerPixel, verticalPixels * orbitRadiansPerPixel);
                }
            }
            previousDragCursor = cursor;
        } else {
            previousDragCursor.reset();
        }

        const auto scrollOffset = inputWindow.consumeScrollYOffset();
        if (!captureMouse) {
            camera.zoom(static_cast<float>(scrollOffset));
        }

        const auto resetPressed = inputWindow.isPressed(GlfwKey::R);
        if (!overlay.wantsKeyboardCapture() && resetPressed && !resetKeyWasPressed) {
            camera.reset();
            previousDragCursor.reset();
        }
        resetKeyWasPressed = resetPressed;
    }

    void drawControlPanel() {
        const auto previousResolution = parameters.resolution;
        const auto previousIsoLevel = parameters.isoLevel;
        const auto *viewport = ImGui::GetMainViewport();
        constexpr float margin = 12.0f;
        const auto panelWidth = std::max(1.0f, std::min(310.0f, viewport->WorkSize.x - 2.0f * margin));
        const auto panelHeight = std::max(1.0f, std::min(560.0f, viewport->WorkSize.y - 2.0f * margin));
        ImGui::SetNextWindowPos({viewport->WorkPos.x + margin, viewport->WorkPos.y + margin}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({panelWidth, panelHeight}, ImGuiCond_Always);

        constexpr ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings;
        if (ImGui::Begin("GPU Marching Cubes", nullptr, windowFlags)) {
            ImGui::SeparatorText("Surface");
            sliderInt("Resolution", "##resolution", parameters.resolution,
                      static_cast<int>(minimumResolution), static_cast<int>(maximumResolution));
            sliderFloat("Iso Level", "##iso_level", parameters.isoLevel, 0.55f, 1.8f, "%.2f");
            sliderFloat("Animation Speed", "##animation_speed", parameters.animationSpeed, 0.0f, 2.0f, "%.2fx");
            ImGui::Checkbox("Pause", &parameters.paused);
            ImGui::SameLine();
            ImGui::Checkbox("Wireframe", &parameters.wireframe);

            ImGui::SeparatorText("Volume Slice");
            ImGui::Checkbox("Show Preview", &parameters.showSlice);
            ImGui::TextUnformatted("Axis");
            ImGui::RadioButton("X", &parameters.sliceAxis, 0);
            ImGui::SameLine();
            ImGui::RadioButton("Y", &parameters.sliceAxis, 1);
            ImGui::SameLine();
            ImGui::RadioButton("Z", &parameters.sliceAxis, 2);
            sliderInt("Slice Index", "##slice_index", parameters.sliceIndex, 0, parameters.resolution);

            ImGui::SeparatorText("Generation");
            if (statistics.valid) {
                const auto triangleCount = statistics.vertexCount / 3;
                const auto cellCount = statistics.resolution * statistics.resolution * statistics.resolution;
                ImGui::Text("Grid: %u^3 (%u cells)", statistics.resolution, cellCount);
                ImGui::Text("Vertices: %u", statistics.vertexCount);
                ImGui::Text("Triangles: %u", triangleCount);
                if (statistics.timingsValid) {
                    ImGui::Text("Volume: %.3f ms", statistics.volumeMilliseconds);
                    ImGui::Text("Extraction: %.3f ms", statistics.extractionMilliseconds);
                } else {
                    ImGui::TextUnformatted("GPU timing: N/A");
                }
            } else {
                ImGui::TextUnformatted("Waiting for GPU results...");
            }

            ImGui::Spacing();
            if (ImGui::Button("Reset", {-1.0f, 0.0f})) {
                parameters = {};
                simulationTime = 0.0f;
                camera.reset();
                previousDragCursor.reset();
                fieldDirty = true;
                meshDirty = true;
            }
        }
        ImGui::End();

        if (parameters.resolution != previousResolution) {
            parameters.sliceIndex = std::clamp(parameters.sliceIndex, 0, parameters.resolution);
            fieldDirty = true;
            meshDirty = true;
        }
        if (parameters.isoLevel != previousIsoLevel) {
            meshDirty = true;
        }
    }

    static void sliderFloat(const char *label,
                            const char *id,
                            float &value,
                            float minimum,
                            float maximum,
                            const char *format) {
        ImGui::TextUnformatted(label);
        ImGui::SetNextItemWidth(-1.0f);
        ImGui::SliderFloat(id, &value, minimum, maximum, format, ImGuiSliderFlags_AlwaysClamp);
    }

    static void sliderInt(const char *label, const char *id, int &value, int minimum, int maximum) {
        ImGui::TextUnformatted(label);
        ImGui::SetNextItemWidth(-1.0f);
        ImGui::SliderInt(id, &value, minimum, maximum, "%d", ImGuiSliderFlags_AlwaysClamp);
    }

    [[nodiscard]] std::uint32_t imguiMinImageCount() const {
        const auto capabilities = physicalDevice().getSurfaceCapabilitiesKHR(*surface());
        return std::max(2u, capabilities.minImageCount);
    }

    [[nodiscard]] ImGuiVulkanOverlayConfig makeOverlayConfig() {
        return {
            .window = &window(),
            .apiVersion = apiVersion(),
            .instance = *instance(),
            .physicalDevice = *physicalDevice(),
            .device = *device(),
            .queueFamilyIndex = graphicsQueueFamilyIndex(),
            .queue = *graphicsQueue(),
            .renderPass = swapchainRenderPass(),
            .minImageCount = imguiMinImageCount(),
            .imageCount = static_cast<std::uint32_t>(swapchainImages().size()),
        };
    }
};

int main() {
    try {
        MarchingCubesApplication application;
        application.run();
    } catch (const std::exception &error) {
        std::println("Error: {}", error.what());
        return 1;
    }
    return 0;
}
