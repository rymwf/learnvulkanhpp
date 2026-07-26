#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>

import vulkan;
import std;
import common;

#ifndef SHADER_SPV_DIR
#error "SHADER_SPV_DIR must be set by CMake (see examples/shadow_vsm/CMakeLists.txt)"
#endif

namespace {

constexpr std::uint32_t shadowMapSize = 1024;
constexpr float cameraVerticalFovRadians = 0.78539816339744830962f;
constexpr float orbitRadiansPerPixel = 0.005f;

struct Vertex {
    std::array<float, 3> position;
    std::array<float, 3> normal;
    std::array<float, 3> color;
};

static_assert(sizeof(Vertex) == 9 * sizeof(float));

struct BufferResource {
    vk::raii::DeviceMemory memory = nullptr;
    vk::raii::Buffer buffer = nullptr;
};

struct ShadowMapResource {
    vk::raii::DeviceMemory memory = nullptr;
    vk::raii::Image image = nullptr;
    vk::raii::ImageView imageView = nullptr;
};

struct ShadowPassResources {
    vk::raii::RenderPass renderPass = nullptr;
    vk::raii::Framebuffer framebuffer = nullptr;
};

struct ColorImageResource {
    vk::raii::DeviceMemory memory = nullptr;
    vk::raii::Image image = nullptr;
    vk::raii::ImageView imageView = nullptr;
};

struct MomentMapResource {
    ColorImageResource raw;
    ColorImageResource filtered;
    vk::raii::Sampler rawSampler = nullptr;
    vk::raii::Sampler filteredSampler = nullptr;
};

struct BlurPassResources {
    vk::raii::RenderPass renderPass = nullptr;
    vk::raii::Framebuffer framebuffer = nullptr;
};

struct DepthPushConstants {
    glm::mat4 lightViewProjection{1.0f};
};

struct ScenePushConstants {
    glm::mat4 cameraViewProjection{1.0f};
    glm::mat4 lightViewProjection{1.0f};
};

static_assert(sizeof(DepthPushConstants) == 16 * sizeof(float));
static_assert(sizeof(ScenePushConstants) == 32 * sizeof(float));

constexpr Vertex makeVertex(float px,
                            float py,
                            float pz,
                            float nx,
                            float ny,
                            float nz,
                            float r,
                            float g,
                            float b) {
    return {
        {px, py, pz},
        {nx, ny, nz},
        {r, g, b},
    };
}

constexpr std::array<Vertex, 28> sceneVertices = {
    // Ground plane.
    makeVertex(-4.0f, 0.0f, -4.0f, 0.0f, 1.0f, 0.0f, 0.45f, 0.50f, 0.43f),
    makeVertex(-4.0f, 0.0f, 4.0f, 0.0f, 1.0f, 0.0f, 0.45f, 0.50f, 0.43f),
    makeVertex(4.0f, 0.0f, 4.0f, 0.0f, 1.0f, 0.0f, 0.45f, 0.50f, 0.43f),
    makeVertex(4.0f, 0.0f, -4.0f, 0.0f, 1.0f, 0.0f, 0.45f, 0.50f, 0.43f),

    // +X face.
    makeVertex(0.75f, 0.0f, -0.75f, 1.0f, 0.0f, 0.0f, 0.82f, 0.34f, 0.12f),
    makeVertex(0.75f, 1.5f, -0.75f, 1.0f, 0.0f, 0.0f, 0.82f, 0.34f, 0.12f),
    makeVertex(0.75f, 1.5f, 0.75f, 1.0f, 0.0f, 0.0f, 0.82f, 0.34f, 0.12f),
    makeVertex(0.75f, 0.0f, 0.75f, 1.0f, 0.0f, 0.0f, 0.82f, 0.34f, 0.12f),

    // -X face.
    makeVertex(-0.75f, 0.0f, 0.75f, -1.0f, 0.0f, 0.0f, 0.82f, 0.34f, 0.12f),
    makeVertex(-0.75f, 1.5f, 0.75f, -1.0f, 0.0f, 0.0f, 0.82f, 0.34f, 0.12f),
    makeVertex(-0.75f, 1.5f, -0.75f, -1.0f, 0.0f, 0.0f, 0.82f, 0.34f, 0.12f),
    makeVertex(-0.75f, 0.0f, -0.75f, -1.0f, 0.0f, 0.0f, 0.82f, 0.34f, 0.12f),

    // +Y face.
    makeVertex(-0.75f, 1.5f, -0.75f, 0.0f, 1.0f, 0.0f, 0.82f, 0.34f, 0.12f),
    makeVertex(-0.75f, 1.5f, 0.75f, 0.0f, 1.0f, 0.0f, 0.82f, 0.34f, 0.12f),
    makeVertex(0.75f, 1.5f, 0.75f, 0.0f, 1.0f, 0.0f, 0.82f, 0.34f, 0.12f),
    makeVertex(0.75f, 1.5f, -0.75f, 0.0f, 1.0f, 0.0f, 0.82f, 0.34f, 0.12f),

    // -Y face.
    makeVertex(-0.75f, 0.0f, 0.75f, 0.0f, -1.0f, 0.0f, 0.82f, 0.34f, 0.12f),
    makeVertex(-0.75f, 0.0f, -0.75f, 0.0f, -1.0f, 0.0f, 0.82f, 0.34f, 0.12f),
    makeVertex(0.75f, 0.0f, -0.75f, 0.0f, -1.0f, 0.0f, 0.82f, 0.34f, 0.12f),
    makeVertex(0.75f, 0.0f, 0.75f, 0.0f, -1.0f, 0.0f, 0.82f, 0.34f, 0.12f),

    // +Z face.
    makeVertex(-0.75f, 0.0f, 0.75f, 0.0f, 0.0f, 1.0f, 0.82f, 0.34f, 0.12f),
    makeVertex(0.75f, 0.0f, 0.75f, 0.0f, 0.0f, 1.0f, 0.82f, 0.34f, 0.12f),
    makeVertex(0.75f, 1.5f, 0.75f, 0.0f, 0.0f, 1.0f, 0.82f, 0.34f, 0.12f),
    makeVertex(-0.75f, 1.5f, 0.75f, 0.0f, 0.0f, 1.0f, 0.82f, 0.34f, 0.12f),

    // -Z face.
    makeVertex(0.75f, 0.0f, -0.75f, 0.0f, 0.0f, -1.0f, 0.82f, 0.34f, 0.12f),
    makeVertex(-0.75f, 0.0f, -0.75f, 0.0f, 0.0f, -1.0f, 0.82f, 0.34f, 0.12f),
    makeVertex(-0.75f, 1.5f, -0.75f, 0.0f, 0.0f, -1.0f, 0.82f, 0.34f, 0.12f),
    makeVertex(0.75f, 1.5f, -0.75f, 0.0f, 0.0f, -1.0f, 0.82f, 0.34f, 0.12f),
};

constexpr std::array<std::uint32_t, 42> sceneIndices = {
    0, 1, 2, 0, 2, 3,
    4, 5, 6, 4, 6, 7,
    8, 9, 10, 8, 10, 11,
    12, 13, 14, 12, 14, 15,
    16, 17, 18, 16, 18, 19,
    20, 21, 22, 20, 22, 23,
    24, 25, 26, 24, 26, 27,
};

glm::mat4 makeLightViewProjection() {
    const auto lightDirection = glm::normalize(glm::vec3{0.50f, 1.0f, -0.70f});
    const auto lightPosition = lightDirection * 8.0f;
    const auto view = glm::lookAtRH(lightPosition, glm::vec3{0.0f, 0.5f, 0.0f}, glm::vec3{0.0f, 1.0f, 0.0f});
    const auto projection = glm::orthoRH_ZO(-5.0f, 5.0f, -5.0f, 5.0f, 0.1f, 20.0f);
    return projection * view;
}

}  // namespace

class VarianceShadowMappingApplication final : public WindowedVulkanApp {
public:
    VarianceShadowMappingApplication()
        : WindowedVulkanApp({
              .window = {
                  .width = 1024,
                  .height = 768,
                  .title = "Vulkan Variance Shadow Maps",
              },
              .apiVersion = vk::ApiVersion11,
          }),
          camera(OrbitCameraConfig{
              .target = {0.0f, 0.5f, 0.0f},
              .yawRadians = 2.35f,
              .pitchRadians = 0.45f,
              .distance = 7.0f,
              .minDistance = 3.0f,
              .maxDistance = 14.0f,
          }) {}

private:
    BufferResource vertexBuffer;
    BufferResource indexBuffer;
    std::uint32_t indexCount = 0;

    vk::Format shadowMapFormat = vk::Format::eUndefined;
    vk::Format momentMapFormat = vk::Format::eUndefined;
    ShadowMapResource shadowMap;
    MomentMapResource momentMap;
    ShadowPassResources shadowPass;
    BlurPassResources blurPass;

    vk::raii::DescriptorSetLayout shadowDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool shadowDescriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> shadowDescriptorSets;

    vk::raii::ShaderModule depthVertexShader = nullptr;
    vk::raii::ShaderModule momentFragmentShader = nullptr;
    vk::raii::ShaderModule sceneVertexShader = nullptr;
    vk::raii::ShaderModule sceneFragmentShader = nullptr;
    vk::raii::ShaderModule previewVertexShader = nullptr;
    vk::raii::ShaderModule previewFragmentShader = nullptr;
    vk::raii::ShaderModule blurFragmentShader = nullptr;

    vk::raii::PipelineLayout depthPipelineLayout = nullptr;
    vk::raii::PipelineLayout scenePipelineLayout = nullptr;
    vk::raii::PipelineLayout previewPipelineLayout = nullptr;
    vk::raii::PipelineLayout blurPipelineLayout = nullptr;
    vk::raii::Pipeline depthPipeline = nullptr;
    vk::raii::Pipeline scenePipeline = nullptr;
    vk::raii::Pipeline previewPipeline = nullptr;
    vk::raii::Pipeline blurPipeline = nullptr;

    OrbitCamera camera;
    std::optional<GlfwCursorPosition> previousDragCursor;
    bool resetKeyWasPressed = false;
    vk::ImageLayout shadowMapLayout = vk::ImageLayout::eUndefined;
    vk::ImageLayout rawMomentLayout = vk::ImageLayout::eUndefined;
    vk::ImageLayout filteredMomentLayout = vk::ImageLayout::eUndefined;

    void createApplicationResources() override {
        createGeometryBuffers();
        createShadowMap();
        createMomentMaps();
        createShadowRenderPass();
        createBlurRenderPass();
        createShadowDescriptors();
        createPipelines();
    }

    void recordFrameCommands(vk::raii::CommandBuffer &commandBuffer, std::uint32_t imageIndex) override {
        updateInput();

        const auto extent = swapchainExtent();
        const auto projection = glm::perspectiveRH_ZO(
            cameraVerticalFovRadians, static_cast<float>(extent.width) / static_cast<float>(std::max(extent.height, 1u)),
            0.1f, 20.0f);
        const ScenePushConstants scenePushConstants{
            .cameraViewProjection = projection * camera.viewMatrix(),
            .lightViewProjection = makeLightViewProjection(),
        };
        const DepthPushConstants depthPushConstants{
            .lightViewProjection = scenePushConstants.lightViewProjection,
        };

        transitionShadowMapToDepthAttachment(commandBuffer);
        transitionRawMomentsToColorAttachment(commandBuffer);
        beginShadowRenderPass(commandBuffer);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *depthPipeline);
        bindSceneGeometry(commandBuffer);
        commandBuffer.pushConstants(*depthPipelineLayout, vk::ShaderStageFlagBits::eVertex, 0,
                                    static_cast<std::uint32_t>(sizeof(depthPushConstants)), &depthPushConstants);
        commandBuffer.drawIndexed(indexCount, 1, 0, 0, 0);
        commandBuffer.endRenderPass();
        transitionRawMomentsToShaderRead(commandBuffer);

        transitionFilteredMomentsToColorAttachment(commandBuffer);
        beginBlurRenderPass(commandBuffer);
        const std::array descriptorSetHandles = {*shadowDescriptorSets.front()};
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *blurPipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *blurPipelineLayout, 0,
                                         descriptorSetHandles, {});
        commandBuffer.draw(4, 1, 0, 0);
        commandBuffer.endRenderPass();
        transitionFilteredMomentsToShaderRead(commandBuffer);

        beginSwapchainRenderPass(commandBuffer, imageIndex);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *scenePipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *scenePipelineLayout, 0,
                                         descriptorSetHandles, {});
        bindSceneGeometry(commandBuffer);
        commandBuffer.pushConstants(*scenePipelineLayout, vk::ShaderStageFlagBits::eVertex, 0,
                                    static_cast<std::uint32_t>(sizeof(scenePushConstants)), &scenePushConstants);
        commandBuffer.drawIndexed(indexCount, 1, 0, 0, 0);

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *previewPipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *previewPipelineLayout, 0,
                                         descriptorSetHandles, {});
        commandBuffer.draw(4, 1, 0, 0);
        commandBuffer.endRenderPass();
    }

    void updateInput() {
        auto &inputWindow = window();
        const auto cursor = inputWindow.cursorPosition();
        const auto panning = inputWindow.isPressed(GlfwMouseButton::Middle);
        const auto orbiting = inputWindow.isPressed(GlfwMouseButton::Left);

        if (panning || orbiting) {
            if (previousDragCursor) {
                const auto horizontalPixels = static_cast<float>(cursor.x - previousDragCursor->x);
                const auto verticalPixels = static_cast<float>(cursor.y - previousDragCursor->y);
                if (panning) {
                    const auto extent = swapchainExtent();
                    camera.pan(horizontalPixels, verticalPixels, cameraVerticalFovRadians,
                               static_cast<float>(std::max(extent.height, 1u)));
                } else {
                    camera.orbit(horizontalPixels * orbitRadiansPerPixel,
                                 verticalPixels * orbitRadiansPerPixel);
                }
            }
            previousDragCursor = cursor;
        } else {
            previousDragCursor.reset();
        }

        camera.zoom(static_cast<float>(inputWindow.consumeScrollYOffset()));

        const auto resetPressed = inputWindow.isPressed(GlfwKey::R);
        if (resetPressed && !resetKeyWasPressed) {
            camera.reset();
            previousDragCursor.reset();
        }
        resetKeyWasPressed = resetPressed;

    }

    BufferResource createHostVisibleBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage) {
        BufferResource resource;
        resource.buffer = device().createBuffer({
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
        });

        const auto memoryRequirements = resource.buffer.getMemoryRequirements();
        resource.memory = vk::raii::DeviceMemory(
            device(),
            {
                .allocationSize = memoryRequirements.size,
                .memoryTypeIndex = findMemoryTypeIndex(
                    memoryRequirements,
                    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent),
            });
        resource.buffer.bindMemory(*resource.memory, 0);
        return resource;
    }

    void uploadBuffer(const BufferResource &resource, const void *source, vk::DeviceSize size) {
        void *mapped = resource.memory.mapMemory(0, size);
        std::memcpy(mapped, source, static_cast<std::size_t>(size));
        resource.memory.unmapMemory();
    }

    void createGeometryBuffers() {
        indexCount = static_cast<std::uint32_t>(sceneIndices.size());
        vertexBuffer = createHostVisibleBuffer(sizeof(sceneVertices), vk::BufferUsageFlagBits::eVertexBuffer);
        uploadBuffer(vertexBuffer, sceneVertices.data(), sizeof(sceneVertices));

        indexBuffer = createHostVisibleBuffer(sizeof(sceneIndices), vk::BufferUsageFlagBits::eIndexBuffer);
        uploadBuffer(indexBuffer, sceneIndices.data(), sizeof(sceneIndices));
    }

    [[nodiscard]] vk::Format findDepthAttachmentFormat() const {
        constexpr std::array candidates = {
            vk::Format::eD32Sfloat,
            vk::Format::eD16Unorm,
        };
        constexpr vk::FormatFeatureFlags requiredFeatures = vk::FormatFeatureFlagBits::eDepthStencilAttachment;
        for (const auto format : candidates) {
            const auto features = physicalDevice().getFormatProperties(format).optimalTilingFeatures;
            if ((features & requiredFeatures) == requiredFeatures) {
                return format;
            }
        }
        throw std::runtime_error("No depth-only format supports VSM depth attachment rendering");
    }

    void createShadowMap() {
        shadowMapFormat = findDepthAttachmentFormat();
        shadowMap.image = device().createImage({
            .imageType = vk::ImageType::e2D,
            .format = shadowMapFormat,
            .extent = {shadowMapSize, shadowMapSize, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined,
        });

        const auto memoryRequirements = shadowMap.image.getMemoryRequirements();
        shadowMap.memory = vk::raii::DeviceMemory(
            device(),
            {
                .allocationSize = memoryRequirements.size,
                .memoryTypeIndex = findMemoryTypeIndex(memoryRequirements, vk::MemoryPropertyFlagBits::eDeviceLocal),
            });
        shadowMap.image.bindMemory(*shadowMap.memory, 0);

        shadowMap.imageView = device().createImageView({
            .image = *shadowMap.image,
            .viewType = vk::ImageViewType::e2D,
            .format = shadowMapFormat,
            .subresourceRange =
                {
                    .aspectMask = vk::ImageAspectFlagBits::eDepth,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
        });

    }

    [[nodiscard]] vk::Format findMomentMapFormat() const {
        constexpr std::array candidates = {
            vk::Format::eR32G32Sfloat,
            vk::Format::eR16G16Sfloat,
        };
        constexpr vk::FormatFeatureFlags requiredFeatures =
            vk::FormatFeatureFlagBits::eColorAttachment | vk::FormatFeatureFlagBits::eSampledImage;
        for (const auto format : candidates) {
            const auto features = physicalDevice().getFormatProperties(format).optimalTilingFeatures;
            if ((features & requiredFeatures) == requiredFeatures) {
                return format;
            }
        }
        throw std::runtime_error("No floating-point color format supports VSM rendering and sampled reads");
    }

    [[nodiscard]] ColorImageResource createMomentImage() {
        ColorImageResource resource;
        resource.image = device().createImage({
            .imageType = vk::ImageType::e2D,
            .format = momentMapFormat,
            .extent = {shadowMapSize, shadowMapSize, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        const auto memoryRequirements = resource.image.getMemoryRequirements();
        resource.memory = vk::raii::DeviceMemory(
            device(),
            {
                .allocationSize = memoryRequirements.size,
                .memoryTypeIndex = findMemoryTypeIndex(memoryRequirements, vk::MemoryPropertyFlagBits::eDeviceLocal),
            });
        resource.image.bindMemory(*resource.memory, 0);
        resource.imageView = device().createImageView({
            .image = *resource.image,
            .viewType = vk::ImageViewType::e2D,
            .format = momentMapFormat,
            .subresourceRange =
                {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
        });
        return resource;
    }

    void createMomentMaps() {
        momentMapFormat = findMomentMapFormat();
        momentMap.raw = createMomentImage();
        momentMap.filtered = createMomentImage();
        const vk::SamplerCreateInfo samplerInfo{
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eNearest,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
            .minLod = 0.0f,
            .maxLod = 0.0f,
        };
        momentMap.rawSampler = device().createSampler(samplerInfo);
        momentMap.filteredSampler = device().createSampler(samplerInfo);
    }

    void createShadowRenderPass() {
        const std::array attachments = {
            vk::AttachmentDescription{
                .format = momentMapFormat,
                .samples = vk::SampleCountFlagBits::e1,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
                .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
                .initialLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
            },
            vk::AttachmentDescription{
                .format = shadowMapFormat,
                .samples = vk::SampleCountFlagBits::e1,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eDontCare,
                .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
                .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
                .initialLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            },
        };
        const vk::AttachmentReference colorReference{
            .attachment = 0,
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
        };
        const vk::AttachmentReference depthReference{
            .attachment = 1,
            .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        };
        const vk::SubpassDescription subpass{
            .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorReference,
            .pDepthStencilAttachment = &depthReference,
        };
        shadowPass.renderPass = device().createRenderPass({
            .attachmentCount = static_cast<std::uint32_t>(attachments.size()),
            .pAttachments = attachments.data(),
            .subpassCount = 1,
            .pSubpasses = &subpass,
        });

        const std::array attachmentsForFramebuffer = {
            *momentMap.raw.imageView,
            *shadowMap.imageView,
        };
        shadowPass.framebuffer = device().createFramebuffer({
            .renderPass = *shadowPass.renderPass,
            .attachmentCount = static_cast<std::uint32_t>(attachmentsForFramebuffer.size()),
            .pAttachments = attachmentsForFramebuffer.data(),
            .width = shadowMapSize,
            .height = shadowMapSize,
            .layers = 1,
        });
    }

    void createBlurRenderPass() {
        const vk::AttachmentDescription attachment{
            .format = momentMapFormat,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
        };
        const vk::AttachmentReference colorReference{
            .attachment = 0,
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
        };
        const vk::SubpassDescription subpass{
            .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorReference,
        };
        blurPass.renderPass = device().createRenderPass({
            .attachmentCount = 1,
            .pAttachments = &attachment,
            .subpassCount = 1,
            .pSubpasses = &subpass,
        });
        const vk::ImageView attachmentView = *momentMap.filtered.imageView;
        blurPass.framebuffer = device().createFramebuffer({
            .renderPass = *blurPass.renderPass,
            .attachmentCount = 1,
            .pAttachments = &attachmentView,
            .width = shadowMapSize,
            .height = shadowMapSize,
            .layers = 1,
        });
    }

    void createShadowDescriptors() {
        const std::array bindings = {
            vk::DescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
            },
        };
        shadowDescriptorSetLayout = device().createDescriptorSetLayout({
            .bindingCount = static_cast<std::uint32_t>(bindings.size()),
            .pBindings = bindings.data(),
        });

        const vk::DescriptorPoolSize poolSize{
            .type = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 2,
        };
        shadowDescriptorPool = device().createDescriptorPool({
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &poolSize,
        });

        const vk::DescriptorSetLayout layout = *shadowDescriptorSetLayout;
        shadowDescriptorSets = device().allocateDescriptorSets({
            .descriptorPool = *shadowDescriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &layout,
        });

        const vk::DescriptorImageInfo rawMomentImageInfo{
            .sampler = *momentMap.rawSampler,
            .imageView = *momentMap.raw.imageView,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };
        const vk::DescriptorImageInfo filteredMomentImageInfo{
            .sampler = *momentMap.filteredSampler,
            .imageView = *momentMap.filtered.imageView,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };
        const std::array writes = {
            vk::WriteDescriptorSet{
                .dstSet = *shadowDescriptorSets.front(),
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &rawMomentImageInfo,
            },
            vk::WriteDescriptorSet{
                .dstSet = *shadowDescriptorSets.front(),
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &filteredMomentImageInfo,
            },
        };
        device().updateDescriptorSets(writes, {});
    }

    [[nodiscard]] vk::raii::ShaderModule loadShaderModule(const std::filesystem::path &path) const {
        const auto spirv = readSpirvFile(path);
        return device().createShaderModule({
            .codeSize = spirv.size() * sizeof(std::uint32_t),
            .pCode = spirv.data(),
        });
    }

    void createPipelines() {
        const std::filesystem::path shaderDirectory{SHADER_SPV_DIR};
        depthVertexShader = loadShaderModule(shaderDirectory / "shadow_depth.vert.spv");
        momentFragmentShader = loadShaderModule(shaderDirectory / "shadow_moments.frag.spv");
        sceneVertexShader = loadShaderModule(shaderDirectory / "shadow_scene.vert.spv");
        sceneFragmentShader = loadShaderModule(shaderDirectory / "shadow_scene.frag.spv");
        previewVertexShader = loadShaderModule(shaderDirectory / "shadow_preview.vert.spv");
        previewFragmentShader = loadShaderModule(shaderDirectory / "shadow_preview.frag.spv");
        blurFragmentShader = loadShaderModule(shaderDirectory / "shadow_blur.frag.spv");

        createDepthPipeline();
        createScenePipeline();
        createBlurPipeline();
        createPreviewPipeline();
    }

    [[nodiscard]] static vk::PipelineVertexInputStateCreateInfo
    makeSceneVertexInput(const vk::VertexInputBindingDescription &bindingDescription,
                         const std::array<vk::VertexInputAttributeDescription, 3> &attributeDescriptions) {
        return {
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions = attributeDescriptions.data(),
        };
    }

    void createDepthPipeline() {
        const std::array shaderStages = {
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = *depthVertexShader,
                .pName = "main",
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = *momentFragmentShader,
                .pName = "main",
            },
        };
        const vk::VertexInputBindingDescription bindingDescription{
            .binding = 0,
            .stride = static_cast<std::uint32_t>(sizeof(Vertex)),
            .inputRate = vk::VertexInputRate::eVertex,
        };
        const vk::VertexInputAttributeDescription positionAttribute{
            .location = 0,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = 0,
        };
        const vk::PipelineVertexInputStateCreateInfo vertexInput{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = 1,
            .pVertexAttributeDescriptions = &positionAttribute,
        };
        const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::eTriangleList,
        };
        const vk::Viewport viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(shadowMapSize),
            .height = static_cast<float>(shadowMapSize),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        const vk::Rect2D scissor{
            .offset = {0, 0},
            .extent = {shadowMapSize, shadowMapSize},
        };
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
            .depthBiasEnable = true,
            .depthBiasConstantFactor = 1.25f,
            .depthBiasSlopeFactor = 1.75f,
            .lineWidth = 1.0f,
        };
        const vk::PipelineMultisampleStateCreateInfo multisampling{};
        const vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = true,
            .depthWriteEnable = true,
            .depthCompareOp = vk::CompareOp::eLessOrEqual,
        };
        const vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG,
        };
        const vk::PipelineColorBlendStateCreateInfo colorBlending{
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment,
        };
        const vk::PushConstantRange pushConstantRange{
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
            .offset = 0,
            .size = static_cast<std::uint32_t>(sizeof(DepthPushConstants)),
        };
        depthPipelineLayout = device().createPipelineLayout({
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstantRange,
        });
        depthPipeline = device().createGraphicsPipeline(nullptr, {
                                                             .stageCount = static_cast<std::uint32_t>(shaderStages.size()),
                                                             .pStages = shaderStages.data(),
                                                             .pVertexInputState = &vertexInput,
                                                             .pInputAssemblyState = &inputAssembly,
                                                             .pViewportState = &viewportState,
                                                             .pRasterizationState = &rasterizer,
                                                             .pMultisampleState = &multisampling,
                                                             .pDepthStencilState = &depthStencil,
                                                             .pColorBlendState = &colorBlending,
                                                             .layout = *depthPipelineLayout,
                                                             .renderPass = *shadowPass.renderPass,
                                                             .subpass = 0,
                                                         });
    }

    void createScenePipeline() {
        const std::array shaderStages = {
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = *sceneVertexShader,
                .pName = "main",
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = *sceneFragmentShader,
                .pName = "main",
            },
        };
        const vk::VertexInputBindingDescription bindingDescription{
            .binding = 0,
            .stride = static_cast<std::uint32_t>(sizeof(Vertex)),
            .inputRate = vk::VertexInputRate::eVertex,
        };
        const std::array attributeDescriptions = {
            vk::VertexInputAttributeDescription{
                .location = 0,
                .binding = 0,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = 0,
            },
            vk::VertexInputAttributeDescription{
                .location = 1,
                .binding = 0,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = 3 * static_cast<std::uint32_t>(sizeof(float)),
            },
            vk::VertexInputAttributeDescription{
                .location = 2,
                .binding = 0,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = 6 * static_cast<std::uint32_t>(sizeof(float)),
            },
        };
        const auto vertexInput = makeSceneVertexInput(bindingDescription, attributeDescriptions);
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
        const vk::PipelineMultisampleStateCreateInfo multisampling{};
        const vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = true,
            .depthWriteEnable = true,
            .depthCompareOp = vk::CompareOp::eLess,
        };
        const vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
        };
        const vk::PipelineColorBlendStateCreateInfo colorBlending{
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment,
        };
        const vk::PushConstantRange pushConstantRange{
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
            .offset = 0,
            .size = static_cast<std::uint32_t>(sizeof(ScenePushConstants)),
        };
        const vk::DescriptorSetLayout descriptorSetLayout = *shadowDescriptorSetLayout;
        scenePipelineLayout = device().createPipelineLayout({
            .setLayoutCount = 1,
            .pSetLayouts = &descriptorSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstantRange,
        });
        scenePipeline = device().createGraphicsPipeline(nullptr, {
                                                             .stageCount = static_cast<std::uint32_t>(shaderStages.size()),
                                                             .pStages = shaderStages.data(),
                                                             .pVertexInputState = &vertexInput,
                                                             .pInputAssemblyState = &inputAssembly,
                                                             .pViewportState = &viewportState,
                                                             .pRasterizationState = &rasterizer,
                                                             .pMultisampleState = &multisampling,
                                                             .pDepthStencilState = &depthStencil,
                                                             .pColorBlendState = &colorBlending,
                                                             .layout = *scenePipelineLayout,
                                                             .renderPass = swapchainRenderPass(),
                                                             .subpass = 0,
                                                         });
    }

    void createBlurPipeline() {
        const std::array shaderStages = {
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = *previewVertexShader,
                .pName = "main",
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = *blurFragmentShader,
                .pName = "main",
            },
        };
        const vk::PipelineVertexInputStateCreateInfo vertexInput{};
        const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::eTriangleStrip,
        };
        const vk::Viewport viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(shadowMapSize),
            .height = static_cast<float>(shadowMapSize),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        const vk::Rect2D scissor{
            .offset = {0, 0},
            .extent = {shadowMapSize, shadowMapSize},
        };
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
        const vk::PipelineMultisampleStateCreateInfo multisampling{};
        const vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = false,
            .depthWriteEnable = false,
        };
        const vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG,
        };
        const vk::PipelineColorBlendStateCreateInfo colorBlending{
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment,
        };
        const vk::DescriptorSetLayout descriptorSetLayout = *shadowDescriptorSetLayout;
        blurPipelineLayout = device().createPipelineLayout({
            .setLayoutCount = 1,
            .pSetLayouts = &descriptorSetLayout,
        });
        blurPipeline = device().createGraphicsPipeline(nullptr, {
                                                            .stageCount = static_cast<std::uint32_t>(shaderStages.size()),
                                                            .pStages = shaderStages.data(),
                                                            .pVertexInputState = &vertexInput,
                                                            .pInputAssemblyState = &inputAssembly,
                                                            .pViewportState = &viewportState,
                                                            .pRasterizationState = &rasterizer,
                                                            .pMultisampleState = &multisampling,
                                                            .pDepthStencilState = &depthStencil,
                                                            .pColorBlendState = &colorBlending,
                                                            .layout = *blurPipelineLayout,
                                                            .renderPass = *blurPass.renderPass,
                                                            .subpass = 0,
                                                        });
    }

    void createPreviewPipeline() {
        const std::array shaderStages = {
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = *previewVertexShader,
                .pName = "main",
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = *previewFragmentShader,
                .pName = "main",
            },
        };
        const vk::PipelineVertexInputStateCreateInfo vertexInput{};
        const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::eTriangleStrip,
        };
        const auto extent = swapchainExtent();
        constexpr std::uint32_t previewSize = 256;
        constexpr std::uint32_t previewMargin = 16;
        const auto previewX = extent.width - previewSize - previewMargin;
        const vk::Viewport viewport{
            .x = 0.0f,
            .y = static_cast<float>(extent.height),
            .width = static_cast<float>(extent.width),
            .height = -static_cast<float>(extent.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        const vk::Rect2D scissor{
            .offset = {static_cast<std::int32_t>(previewX), static_cast<std::int32_t>(previewMargin)},
            .extent = {previewSize, previewSize},
        };
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
        const vk::PipelineMultisampleStateCreateInfo multisampling{};
        const vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = false,
            .depthWriteEnable = false,
        };
        const vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
        };
        const vk::PipelineColorBlendStateCreateInfo colorBlending{
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment,
        };
        const vk::DescriptorSetLayout descriptorSetLayout = *shadowDescriptorSetLayout;
        previewPipelineLayout = device().createPipelineLayout({
            .setLayoutCount = 1,
            .pSetLayouts = &descriptorSetLayout,
        });
        previewPipeline = device().createGraphicsPipeline(nullptr, {
                                                               .stageCount = static_cast<std::uint32_t>(shaderStages.size()),
                                                               .pStages = shaderStages.data(),
                                                               .pVertexInputState = &vertexInput,
                                                               .pInputAssemblyState = &inputAssembly,
                                                               .pViewportState = &viewportState,
                                                               .pRasterizationState = &rasterizer,
                                                               .pMultisampleState = &multisampling,
                                                               .pDepthStencilState = &depthStencil,
                                                               .pColorBlendState = &colorBlending,
                                                               .layout = *previewPipelineLayout,
                                                               .renderPass = swapchainRenderPass(),
                                                               .subpass = 0,
                                                           });
    }

    void bindSceneGeometry(vk::raii::CommandBuffer &commandBuffer) const {
        const vk::Buffer rawVertexBuffer = *vertexBuffer.buffer;
        constexpr vk::DeviceSize vertexOffset = 0;
        commandBuffer.bindVertexBuffers(0, {rawVertexBuffer}, {vertexOffset});
        commandBuffer.bindIndexBuffer(*indexBuffer.buffer, 0, vk::IndexType::eUint32);
    }

    void beginShadowRenderPass(vk::raii::CommandBuffer &commandBuffer) const {
        std::array clearValues = {vk::ClearValue{}, vk::ClearValue{}};
        clearValues[0].color.setFloat32(std::array<float, 4>{1.0f, 1.0f, 0.0f, 1.0f});
        clearValues[1].depthStencil = vk::ClearDepthStencilValue{1.0f, 0};
        const vk::RenderPassBeginInfo beginInfo{
            .renderPass = *shadowPass.renderPass,
            .framebuffer = *shadowPass.framebuffer,
            .renderArea = {
                .offset = {0, 0},
                .extent = {shadowMapSize, shadowMapSize},
            },
            .clearValueCount = static_cast<std::uint32_t>(clearValues.size()),
            .pClearValues = clearValues.data(),
        };
        commandBuffer.beginRenderPass(beginInfo, vk::SubpassContents::eInline);
    }

    void beginBlurRenderPass(vk::raii::CommandBuffer &commandBuffer) const {
        vk::ClearValue clearValue{};
        clearValue.color.setFloat32(std::array<float, 4>{1.0f, 1.0f, 0.0f, 1.0f});
        const vk::RenderPassBeginInfo beginInfo{
            .renderPass = *blurPass.renderPass,
            .framebuffer = *blurPass.framebuffer,
            .renderArea = {
                .offset = {0, 0},
                .extent = {shadowMapSize, shadowMapSize},
            },
            .clearValueCount = 1,
            .pClearValues = &clearValue,
        };
        commandBuffer.beginRenderPass(beginInfo, vk::SubpassContents::eInline);
    }

    void transitionShadowMapToDepthAttachment(vk::raii::CommandBuffer &commandBuffer) {
        if (shadowMapLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
            return;
        }
        transitionImage(commandBuffer, *shadowMap.image, vk::ImageAspectFlagBits::eDepth, shadowMapLayout,
                        vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::PipelineStageFlagBits::eTopOfPipe,
                        vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests,
                        {}, vk::AccessFlagBits::eDepthStencilAttachmentRead |
                                vk::AccessFlagBits::eDepthStencilAttachmentWrite);
        shadowMapLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    }

    void transitionRawMomentsToColorAttachment(vk::raii::CommandBuffer &commandBuffer) {
        const auto sourceStage = rawMomentLayout == vk::ImageLayout::eUndefined
                                     ? vk::PipelineStageFlags{vk::PipelineStageFlagBits::eTopOfPipe}
                                     : vk::PipelineStageFlags{vk::PipelineStageFlagBits::eFragmentShader};
        const auto sourceAccess = rawMomentLayout == vk::ImageLayout::eUndefined
                                      ? vk::AccessFlags{}
                                      : vk::AccessFlags{vk::AccessFlagBits::eShaderRead};
        transitionImage(commandBuffer, *momentMap.raw.image, vk::ImageAspectFlagBits::eColor, rawMomentLayout,
                        vk::ImageLayout::eColorAttachmentOptimal, sourceStage,
                        vk::PipelineStageFlagBits::eColorAttachmentOutput, sourceAccess,
                        vk::AccessFlagBits::eColorAttachmentWrite);
        rawMomentLayout = vk::ImageLayout::eColorAttachmentOptimal;
    }

    void transitionRawMomentsToShaderRead(vk::raii::CommandBuffer &commandBuffer) {
        transitionImage(commandBuffer, *momentMap.raw.image, vk::ImageAspectFlagBits::eColor,
                        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader,
                        vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eShaderRead);
        rawMomentLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    }

    void transitionFilteredMomentsToColorAttachment(vk::raii::CommandBuffer &commandBuffer) {
        const auto sourceStage = filteredMomentLayout == vk::ImageLayout::eUndefined
                                     ? vk::PipelineStageFlags{vk::PipelineStageFlagBits::eTopOfPipe}
                                     : vk::PipelineStageFlags{vk::PipelineStageFlagBits::eFragmentShader};
        const auto sourceAccess = filteredMomentLayout == vk::ImageLayout::eUndefined
                                      ? vk::AccessFlags{}
                                      : vk::AccessFlags{vk::AccessFlagBits::eShaderRead};
        transitionImage(commandBuffer, *momentMap.filtered.image, vk::ImageAspectFlagBits::eColor, filteredMomentLayout,
                        vk::ImageLayout::eColorAttachmentOptimal, sourceStage,
                        vk::PipelineStageFlagBits::eColorAttachmentOutput, sourceAccess,
                        vk::AccessFlagBits::eColorAttachmentWrite);
        filteredMomentLayout = vk::ImageLayout::eColorAttachmentOptimal;
    }

    void transitionFilteredMomentsToShaderRead(vk::raii::CommandBuffer &commandBuffer) {
        transitionImage(commandBuffer, *momentMap.filtered.image, vk::ImageAspectFlagBits::eColor,
                        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader,
                        vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eShaderRead);
        filteredMomentLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    }

    void transitionImage(vk::raii::CommandBuffer &commandBuffer,
                         vk::Image image,
                         vk::ImageAspectFlags aspectMask,
                         vk::ImageLayout oldLayout,
                         vk::ImageLayout newLayout,
                         vk::PipelineStageFlags sourceStage,
                         vk::PipelineStageFlags destinationStage,
                         vk::AccessFlags sourceAccess,
                         vk::AccessFlags destinationAccess) {
        const vk::ImageMemoryBarrier barrier{
            .srcAccessMask = sourceAccess,
            .dstAccessMask = destinationAccess,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = std::numeric_limits<std::uint32_t>::max(),
            .dstQueueFamilyIndex = std::numeric_limits<std::uint32_t>::max(),
            .image = image,
            .subresourceRange =
                {
                    .aspectMask = aspectMask,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
        };
        commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, {barrier});
    }
};

int main() {
    try {
        VarianceShadowMappingApplication application;
        application.run();
    } catch (const std::exception &error) {
        std::println("Error: {}", error.what());
        return 1;
    }
    return 0;
}
