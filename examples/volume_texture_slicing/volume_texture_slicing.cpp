#include <glm/ext/matrix_clip_space.hpp>
#include <glm/glm.hpp>
#include <imgui.h>

import vulkan;
import std;
import common;
import common.imgui_overlay;

#ifndef SHADER_SPV_DIR
#error "SHADER_SPV_DIR must be set by CMake (see examples/volume_texture_slicing/CMakeLists.txt)"
#endif

namespace {

constexpr std::uint32_t volumeResolution = 96;
constexpr vk::Format volumeFormat = vk::Format::eR32Sfloat;
constexpr std::uint32_t computeLocalSize = 4;
constexpr std::uint32_t queriesPerFrame = 3;
constexpr float cameraVerticalFovRadians = 0.78539816339744830962f;
constexpr float orbitRadiansPerPixel = 0.005f;
constexpr glm::vec3 worldUp{0.0f, 1.0f, 0.0f};

struct ImageResource {
    vk::raii::DeviceMemory memory = nullptr;
    vk::raii::Image image = nullptr;
    vk::raii::ImageView imageView = nullptr;
};

struct alignas(16) FieldPushConstants {
    float time = 0.0f;
    std::uint32_t resolution = volumeResolution;
    float padding0 = 0.0f;
    float padding1 = 0.0f;
};

struct alignas(16) SlicePushConstants {
    glm::mat4 viewProjection{1.0f};
    glm::vec4 cameraRightAndExtent{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec4 cameraUpAndSliceCount{0.0f, 1.0f, 0.0f, 128.0f};
    glm::vec4 cameraForwardAndOpacity{0.0f, 0.0f, -1.0f, 1.0f};
    glm::vec4 densityAndShading{0.25f, 1.75f, 1.0f, 0.0f};
};

struct Parameters {
    int sliceCount = 128;
    float densityMinimum = 0.25f;
    float densityMaximum = 1.75f;
    float opacityScale = 1.0f;
    float animationSpeed = 1.0f;
    bool gradientShading = true;
    bool paused = false;
};

struct TimingSlot {
    bool pending = false;
    bool generatedVolume = false;
};

struct TimingStatistics {
    bool valid = false;
    bool generatedVolume = false;
    double generationMilliseconds = 0.0;
    double slicingMilliseconds = 0.0;
};

static_assert(sizeof(FieldPushConstants) == 16);
static_assert(sizeof(SlicePushConstants) == 128);

[[nodiscard]] std::uint32_t dispatchGroupCount(std::uint32_t invocationCount) {
    return (invocationCount + computeLocalSize - 1u) / computeLocalSize;
}

}  // namespace

class VolumeTextureSlicingApplication final : public WindowedVulkanApp {
public:
    VolumeTextureSlicingApplication()
        : WindowedVulkanApp({
              .window = {
                  .width = 1120,
                  .height = 760,
                  .title = "Vulkan 3D Texture Volume Slicing",
              },
              .apiVersion = vk::ApiVersion11,
              .requiredQueueFlags = vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute,
          }),
          camera(OrbitCameraConfig{
              .target = {0.0f, 0.0f, 0.0f},
              .yawRadians = 2.35f,
              .pitchRadians = 0.34f,
              .distance = 5.2f,
              .minDistance = 2.8f,
              .maxDistance = 10.0f,
          }) {}

private:
    ImageResource volumeImage;
    vk::raii::Sampler volumeSampler = nullptr;

    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> descriptorSets;

    vk::raii::ShaderModule fieldShader = nullptr;
    vk::raii::ShaderModule sliceVertexShader = nullptr;
    vk::raii::ShaderModule sliceFragmentShader = nullptr;
    vk::raii::PipelineLayout computePipelineLayout = nullptr;
    vk::raii::PipelineLayout slicePipelineLayout = nullptr;
    vk::raii::Pipeline fieldPipeline = nullptr;
    vk::raii::Pipeline slicePipeline = nullptr;

    vk::raii::QueryPool timestampQueryPool = nullptr;
    std::vector<TimingSlot> timingSlots;
    std::uint32_t timestampValidBits = 0;
    double timestampPeriodNanoseconds = 0.0;
    TimingStatistics timingStatistics;

    OrbitCamera camera;
    std::optional<GlfwCursorPosition> previousDragCursor;
    bool resetKeyWasPressed = false;
    Parameters parameters;
    float simulationTime = 0.0f;
    std::chrono::steady_clock::time_point previousFrameTime = std::chrono::steady_clock::now();
    vk::ImageLayout volumeImageLayout = vk::ImageLayout::eUndefined;
    bool fieldDirty = true;
    ImGuiVulkanOverlay overlay;

    bool supportsAdditionalDeviceRequirements(const vk::raii::PhysicalDevice &candidate) const override {
        const auto formatProperties = candidate.getFormatProperties(volumeFormat);
        const auto requiredFormatFeatures = vk::FormatFeatureFlagBits::eStorageImage |
                                            vk::FormatFeatureFlagBits::eSampledImage |
                                            vk::FormatFeatureFlagBits::eSampledImageFilterLinear;
        if ((formatProperties.optimalTilingFeatures & requiredFormatFeatures) != requiredFormatFeatures) {
            return false;
        }
        return candidate.getProperties().limits.maxImageDimension3D >= volumeResolution;
    }

    void createApplicationResources() override {
        createVolumeImage();
        createVolumeSampler();
        createDescriptors();
        createPipelineLayouts();
        createShaderModules();
        createComputePipeline();
        createSlicePipeline();
        createTimingResources();
        overlay.initialize(makeOverlayConfig());
    }

    void releaseSwapchainDependentResources() override {
        overlay.releaseRenderer();
        slicePipeline = nullptr;
    }

    void recreateSwapchainDependentResources() override {
        createSlicePipeline();
        overlay.recreateRenderer(swapchainRenderPass(), imguiMinImageCount(),
                                 static_cast<std::uint32_t>(swapchainImages().size()));
    }

    void recordFrameCommands(vk::raii::CommandBuffer &commandBuffer, std::uint32_t imageIndex) override {
        const auto frameIndex = currentFrameIndex();
        readCompletedTiming(frameIndex);

        overlay.beginFrame();
        drawControlPanel();
        updateInput();
        updateSimulationTime();

        const bool generatedVolume = fieldDirty;
        beginTiming(commandBuffer, frameIndex);
        recordFieldGeneration(commandBuffer);
        markGenerationComplete(commandBuffer, frameIndex, generatedVolume);
        recordSlices(commandBuffer, imageIndex, frameIndex, generatedVolume);
    }

    void createVolumeImage() {
        volumeImage.image = device().createImage({
            .imageType = vk::ImageType::e3D,
            .format = volumeFormat,
            .extent = {volumeResolution, volumeResolution, volumeResolution},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
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

    void createVolumeSampler() {
        volumeSampler = device().createSampler({
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eNearest,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
            .mipLodBias = 0.0f,
            .anisotropyEnable = false,
            .maxAnisotropy = 1.0f,
            .compareEnable = false,
            .compareOp = vk::CompareOp::eAlways,
            .minLod = 0.0f,
            .maxLod = 0.0f,
            .borderColor = vk::BorderColor::eFloatOpaqueBlack,
            .unnormalizedCoordinates = false,
        });
    }

    void createDescriptors() {
        const std::array bindings = {
            vk::DescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = vk::DescriptorType::eStorageImage,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eCompute,
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
            },
        };
        descriptorSetLayout = device().createDescriptorSetLayout({
            .bindingCount = static_cast<std::uint32_t>(bindings.size()),
            .pBindings = bindings.data(),
        });

        const std::array poolSizes = {
            vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageImage, .descriptorCount = 1},
            vk::DescriptorPoolSize{.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1},
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

        const vk::DescriptorImageInfo storageImageInfo{
            .imageView = *volumeImage.imageView,
            .imageLayout = vk::ImageLayout::eGeneral,
        };
        const vk::DescriptorImageInfo sampledImageInfo{
            .sampler = *volumeSampler,
            .imageView = *volumeImage.imageView,
            .imageLayout = vk::ImageLayout::eGeneral,
        };
        const std::array writes = {
            vk::WriteDescriptorSet{
                .dstSet = *descriptorSets.front(),
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageImage,
                .pImageInfo = &storageImageInfo,
            },
            vk::WriteDescriptorSet{
                .dstSet = *descriptorSets.front(),
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &sampledImageInfo,
            },
        };
        device().updateDescriptorSets(writes, {});
    }

    void createPipelineLayouts() {
        const vk::DescriptorSetLayout rawLayout = *descriptorSetLayout;
        const vk::PushConstantRange computeRange{
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset = 0,
            .size = sizeof(FieldPushConstants),
        };
        computePipelineLayout = device().createPipelineLayout({
            .setLayoutCount = 1,
            .pSetLayouts = &rawLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &computeRange,
        });

        const vk::PushConstantRange sliceRange{
            .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
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
        sliceVertexShader = loadShaderModule(shaderDirectory / "slice.vert.spv");
        sliceFragmentShader = loadShaderModule(shaderDirectory / "slice.frag.spv");
    }

    void createComputePipeline() {
        fieldPipeline = device().createComputePipeline(nullptr, {
                                                                    .stage = {
                                                                        .stage = vk::ShaderStageFlagBits::eCompute,
                                                                        .module = *fieldShader,
                                                                        .pName = "main",
                                                                    },
                                                                    .layout = *computePipelineLayout,
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
            .topology = vk::PrimitiveTopology::eTriangleStrip,
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
        const vk::PipelineDepthStencilStateCreateInfo depthStencil{};
        const vk::PipelineColorBlendAttachmentState blendAttachment{
            .blendEnable = true,
            .srcColorBlendFactor = vk::BlendFactor::eOne,
            .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
            .colorBlendOp = vk::BlendOp::eAdd,
            .srcAlphaBlendFactor = vk::BlendFactor::eOne,
            .dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
            .alphaBlendOp = vk::BlendOp::eAdd,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
        };
        const vk::PipelineColorBlendStateCreateInfo blending{
            .attachmentCount = 1,
            .pAttachments = &blendAttachment,
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
                                                                      .layout = *slicePipelineLayout,
                                                                      .renderPass = swapchainRenderPass(),
                                                                      .subpass = 0,
                                                                  });
    }

    void createTimingResources() {
        timingSlots.resize(framesInFlight());
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

    void beginTiming(vk::raii::CommandBuffer &commandBuffer, std::uint32_t frameIndex) {
        if (timestampValidBits == 0) {
            return;
        }
        const auto queryBase = frameIndex * queriesPerFrame;
        commandBuffer.resetQueryPool(*timestampQueryPool, queryBase, queriesPerFrame);
        commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *timestampQueryPool, queryBase);
    }

    void markGenerationComplete(vk::raii::CommandBuffer &commandBuffer,
                                std::uint32_t frameIndex,
                                bool generatedVolume) {
        if (timestampValidBits == 0) {
            return;
        }
        const auto queryBase = frameIndex * queriesPerFrame;
        const auto stage = generatedVolume ? vk::PipelineStageFlagBits::eComputeShader
                                           : vk::PipelineStageFlagBits::eTopOfPipe;
        commandBuffer.writeTimestamp(stage, *timestampQueryPool, queryBase + 1);
    }

    void markSlicingComplete(vk::raii::CommandBuffer &commandBuffer,
                             std::uint32_t frameIndex,
                             bool generatedVolume) {
        if (timestampValidBits == 0) {
            return;
        }
        const auto queryBase = frameIndex * queriesPerFrame;
        commandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eFragmentShader, *timestampQueryPool,
                                     queryBase + 2);
        auto &slot = timingSlots.at(frameIndex);
        slot.pending = true;
        slot.generatedVolume = generatedVolume;
    }

    void readCompletedTiming(std::uint32_t frameIndex) {
        auto &slot = timingSlots.at(frameIndex);
        if (timestampValidBits == 0 || !slot.pending) {
            return;
        }

        std::array<std::uint64_t, queriesPerFrame> timestamps{};
        const auto firstQuery = frameIndex * queriesPerFrame;
        const auto result = timestampQueryPool.getResults(
            firstQuery, queriesPerFrame, sizeof(timestamps), timestamps.data(), sizeof(std::uint64_t),
            vk::QueryResultFlagBits::e64);
        if (result == vk::Result::eSuccess) {
            timingStatistics.valid = true;
            timingStatistics.generatedVolume = slot.generatedVolume;
            timingStatistics.generationMilliseconds =
                slot.generatedVolume ? timestampDelta(timestamps[0], timestamps[1]) : 0.0;
            timingStatistics.slicingMilliseconds = timestampDelta(timestamps[1], timestamps[2]);
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

    void recordFieldGeneration(vk::raii::CommandBuffer &commandBuffer) {
        if (!fieldDirty) {
            return;
        }

        transitionVolumeForWrite(commandBuffer);
        const FieldPushConstants pushConstants{
            .time = simulationTime,
        };
        const std::array descriptorSetHandles = {*descriptorSets.front()};
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *fieldPipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computePipelineLayout, 0,
                                         descriptorSetHandles, {});
        commandBuffer.pushConstants(*computePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                                    sizeof(pushConstants), &pushConstants);
        const auto groups = dispatchGroupCount(volumeResolution);
        commandBuffer.dispatch(groups, groups, groups);
        transitionVolumeForRead(commandBuffer);
        fieldDirty = false;
    }

    void transitionVolumeForWrite(vk::raii::CommandBuffer &commandBuffer) {
        const auto firstUse = volumeImageLayout == vk::ImageLayout::eUndefined;
        const vk::ImageMemoryBarrier barrier{
            .srcAccessMask = firstUse ? vk::AccessFlags{} : vk::AccessFlags{vk::AccessFlagBits::eShaderRead},
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
        const auto sourceStage = firstUse ? vk::PipelineStageFlags{vk::PipelineStageFlagBits::eTopOfPipe}
                                          : vk::PipelineStageFlags{vk::PipelineStageFlagBits::eFragmentShader};
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
                                      vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, {barrier});
    }

    void recordSlices(vk::raii::CommandBuffer &commandBuffer,
                      std::uint32_t imageIndex,
                      std::uint32_t frameIndex,
                      bool generatedVolume) {
        const auto extent = swapchainExtent();
        const auto projection = glm::perspectiveRH_ZO(
            cameraVerticalFovRadians,
            static_cast<float>(extent.width) / static_cast<float>(std::max(extent.height, 1u)), 0.1f, 20.0f);
        const auto forward = glm::normalize(camera.target() - camera.position());
        const auto right = glm::normalize(glm::cross(forward, worldUp));
        const auto up = glm::cross(right, forward);
        const SlicePushConstants pushConstants{
            .viewProjection = projection * camera.viewMatrix(),
            .cameraRightAndExtent = {right, 0.0f},
            .cameraUpAndSliceCount = {up, static_cast<float>(parameters.sliceCount)},
            .cameraForwardAndOpacity = {forward, parameters.opacityScale},
            .densityAndShading = {
                parameters.densityMinimum,
                parameters.densityMaximum,
                parameters.gradientShading ? 1.0f : 0.0f,
                0.0f,
            },
        };
        const std::array descriptorSetHandles = {*descriptorSets.front()};

        std::array<vk::ClearValue, 2> clearValues{};
        clearValues[0].color.setFloat32(std::array<float, 4>{0.018f, 0.026f, 0.038f, 1.0f});
        clearValues[1].depthStencil = vk::ClearDepthStencilValue{1.0f, 0};
        beginSwapchainRenderPass(commandBuffer, imageIndex, clearValues);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *slicePipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *slicePipelineLayout, 0,
                                         descriptorSetHandles, {});
        commandBuffer.pushConstants(*slicePipelineLayout,
                                    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
                                    sizeof(pushConstants), &pushConstants);
        commandBuffer.draw(4, static_cast<std::uint32_t>(parameters.sliceCount), 0, 0);
        markSlicingComplete(commandBuffer, frameIndex, generatedVolume);
        overlay.render(commandBuffer);
        commandBuffer.endRenderPass();
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
        const auto *viewport = ImGui::GetMainViewport();
        constexpr float margin = 12.0f;
        const auto panelWidth = std::max(1.0f, std::min(320.0f, viewport->WorkSize.x - 2.0f * margin));
        const auto panelHeight = std::max(1.0f, std::min(520.0f, viewport->WorkSize.y - 2.0f * margin));
        ImGui::SetNextWindowPos({viewport->WorkPos.x + margin, viewport->WorkPos.y + margin}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({panelWidth, panelHeight}, ImGuiCond_Always);

        constexpr ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings;
        if (ImGui::Begin("3D Texture Volume Slicing", nullptr, windowFlags)) {
            ImGui::SeparatorText("Slicing");
            ImGui::TextUnformatted("Slices");
            ImGui::SetNextItemWidth(-1.0f);
            ImGui::SliderInt("##slice_count", &parameters.sliceCount, 32, 256, "%d",
                             ImGuiSliderFlags_AlwaysClamp);
            sliderFloat("Density Minimum", "##density_minimum", parameters.densityMinimum, 0.05f, 1.5f, "%.2f");
            sliderFloat("Density Maximum", "##density_maximum", parameters.densityMaximum, 0.5f, 3.0f, "%.2f");
            parameters.densityMaximum =
                std::max(parameters.densityMaximum, parameters.densityMinimum + 0.05f);
            parameters.densityMinimum =
                std::min(parameters.densityMinimum, parameters.densityMaximum - 0.05f);
            sliderFloat("Opacity", "##opacity", parameters.opacityScale, 0.1f, 4.0f, "%.2fx");
            ImGui::Checkbox("Gradient Shading", &parameters.gradientShading);

            ImGui::SeparatorText("Animation");
            sliderFloat("Speed", "##animation_speed", parameters.animationSpeed, 0.0f, 2.0f, "%.2fx");
            ImGui::Checkbox("Pause", &parameters.paused);

            ImGui::SeparatorText("GPU");
            ImGui::Text("Volume: %u^3", volumeResolution);
            if (timestampValidBits == 0) {
                ImGui::TextUnformatted("GPU timing: N/A");
            } else if (timingStatistics.valid) {
                if (timingStatistics.generatedVolume) {
                    ImGui::Text("Generation: %.3f ms", timingStatistics.generationMilliseconds);
                } else {
                    ImGui::TextUnformatted("Generation: paused");
                }
                ImGui::Text("Slicing: %.3f ms", timingStatistics.slicingMilliseconds);
            } else {
                ImGui::TextUnformatted("Waiting for GPU timing...");
            }

            ImGui::Spacing();
            if (ImGui::Button("Reset", {-1.0f, 0.0f})) {
                parameters = {};
                simulationTime = 0.0f;
                camera.reset();
                previousDragCursor.reset();
                fieldDirty = true;
            }
        }
        ImGui::End();
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
        VolumeTextureSlicingApplication application;
        application.run();
    } catch (const std::exception &error) {
        std::println("Error: {}", error.what());
        return 1;
    }
    return 0;
}
