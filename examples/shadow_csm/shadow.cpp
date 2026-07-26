#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/glm.hpp>

import vulkan;
import std;
import common;

#ifndef SHADER_SPV_DIR
#error "SHADER_SPV_DIR must be set by CMake (see examples/shadow_csm/CMakeLists.txt)"
#endif

namespace {

constexpr std::uint32_t shadowMapSize = 1024;
constexpr std::uint32_t cascadeCount = 3;
constexpr std::array cascadeSplitDistances = {20.0f, 70.0f, 200.0f};
constexpr float cameraNearClip = 0.5f;
constexpr float cameraFarClip = cascadeSplitDistances.back();
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
    vk::raii::ImageView arrayImageView = nullptr;
    std::vector<vk::raii::ImageView> layerImageViews;
    vk::raii::Sampler comparisonSampler = nullptr;
    vk::raii::Sampler previewSampler = nullptr;
};

struct ShadowPassResources {
    vk::raii::RenderPass renderPass = nullptr;
    std::vector<vk::raii::Framebuffer> framebuffers;
};

struct DepthPushConstants {
    glm::mat4 lightViewProjection{1.0f};
};

struct ScenePushConstants {
    glm::mat4 cameraViewProjection{1.0f};
    glm::mat4 cameraView{1.0f};
};

static_assert(sizeof(DepthPushConstants) == 16 * sizeof(float));
static_assert(sizeof(ScenePushConstants) == 32 * sizeof(float));

struct CascadeUniforms {
    std::array<glm::mat4, cascadeCount> lightViewProjections{};
    glm::vec4 cascadeSplits{};
};

static_assert(sizeof(CascadeUniforms) == 52 * sizeof(float));

struct SceneGeometry {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
};

struct BoxDefinition {
    glm::vec3 minimum;
    glm::vec3 maximum;
    glm::vec3 color;
};

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

void appendFace(SceneGeometry &geometry,
                glm::vec3 first,
                glm::vec3 second,
                glm::vec3 third,
                glm::vec3 fourth,
                glm::vec3 normal,
                glm::vec3 color) {
    const auto baseIndex = static_cast<std::uint32_t>(geometry.vertices.size());
    const auto appendVertex = [&geometry, normal, color](glm::vec3 position) {
        geometry.vertices.push_back(makeVertex(position.x, position.y, position.z, normal.x, normal.y, normal.z,
                                               color.r, color.g, color.b));
    };
    appendVertex(first);
    appendVertex(second);
    appendVertex(third);
    appendVertex(fourth);
    geometry.indices.insert(geometry.indices.end(),
                            {baseIndex, baseIndex + 1, baseIndex + 2, baseIndex, baseIndex + 2, baseIndex + 3});
}

void appendBox(SceneGeometry &geometry, const BoxDefinition &box) {
    const auto &minimum = box.minimum;
    const auto &maximum = box.maximum;
    const auto &color = box.color;

    appendFace(geometry, {maximum.x, minimum.y, minimum.z}, {maximum.x, maximum.y, minimum.z},
               {maximum.x, maximum.y, maximum.z}, {maximum.x, minimum.y, maximum.z}, {1.0f, 0.0f, 0.0f}, color);
    appendFace(geometry, {minimum.x, minimum.y, maximum.z}, {minimum.x, maximum.y, maximum.z},
               {minimum.x, maximum.y, minimum.z}, {minimum.x, minimum.y, minimum.z}, {-1.0f, 0.0f, 0.0f}, color);
    appendFace(geometry, {minimum.x, maximum.y, minimum.z}, {minimum.x, maximum.y, maximum.z},
               {maximum.x, maximum.y, maximum.z}, {maximum.x, maximum.y, minimum.z}, {0.0f, 1.0f, 0.0f}, color);
    appendFace(geometry, {minimum.x, minimum.y, maximum.z}, {minimum.x, minimum.y, minimum.z},
               {maximum.x, minimum.y, minimum.z}, {maximum.x, minimum.y, maximum.z}, {0.0f, -1.0f, 0.0f}, color);
    appendFace(geometry, {minimum.x, minimum.y, maximum.z}, {maximum.x, minimum.y, maximum.z},
               {maximum.x, maximum.y, maximum.z}, {minimum.x, maximum.y, maximum.z}, {0.0f, 0.0f, 1.0f}, color);
    appendFace(geometry, {maximum.x, minimum.y, minimum.z}, {minimum.x, minimum.y, minimum.z},
               {minimum.x, maximum.y, minimum.z}, {maximum.x, maximum.y, minimum.z}, {0.0f, 0.0f, -1.0f}, color);
}

SceneGeometry makeLargeSceneGeometry() {
    const std::array<BoxDefinition, 12> buildings = {
        BoxDefinition{{-7.0f, 0.0f, -16.0f}, {1.0f, 16.0f, -8.0f}, {0.78f, 0.34f, 0.14f}},
        BoxDefinition{{3.0f, 0.0f, -13.0f}, {10.0f, 7.0f, -6.0f}, {0.28f, 0.48f, 0.72f}},
        BoxDefinition{{-4.0f, 0.0f, 17.0f}, {4.0f, 11.0f, 25.0f}, {0.66f, 0.55f, 0.22f}},
        BoxDefinition{{-32.0f, 0.0f, 36.0f}, {-20.0f, 10.0f, 48.0f}, {0.56f, 0.29f, 0.62f}},
        BoxDefinition{{10.0f, 0.0f, 45.0f}, {22.0f, 20.0f, 58.0f}, {0.23f, 0.57f, 0.50f}},
        BoxDefinition{{-7.0f, 0.0f, 63.0f}, {5.0f, 8.0f, 75.0f}, {0.72f, 0.42f, 0.18f}},
        BoxDefinition{{-28.0f, 0.0f, 85.0f}, {-15.0f, 15.0f, 99.0f}, {0.25f, 0.38f, 0.67f}},
        BoxDefinition{{16.0f, 0.0f, 104.0f}, {30.0f, 24.0f, 119.0f}, {0.67f, 0.28f, 0.35f}},
        BoxDefinition{{-7.0f, 0.0f, 125.0f}, {7.0f, 11.0f, 139.0f}, {0.63f, 0.57f, 0.24f}},
        BoxDefinition{{-38.0f, 0.0f, 149.0f}, {-23.0f, 32.0f, 165.0f}, {0.32f, 0.52f, 0.67f}},
        BoxDefinition{{20.0f, 0.0f, 165.0f}, {35.0f, 16.0f, 181.0f}, {0.56f, 0.34f, 0.20f}},
        BoxDefinition{{-8.0f, 0.0f, 177.0f}, {9.0f, 11.0f, 189.0f}, {0.38f, 0.58f, 0.37f}},
    };

    SceneGeometry geometry;
    geometry.vertices.reserve(4 + buildings.size() * 24);
    geometry.indices.reserve(6 + buildings.size() * 36);
    appendFace(geometry, {-100.0f, 0.0f, -30.0f}, {-100.0f, 0.0f, 190.0f}, {100.0f, 0.0f, 190.0f},
               {100.0f, 0.0f, -30.0f}, {0.0f, 1.0f, 0.0f}, {0.39f, 0.46f, 0.36f});
    for (const auto &building : buildings) {
        appendBox(geometry, building);
    }
    return geometry;
}

std::array<glm::mat4, cascadeCount> makeCascadeLightViewProjections(const glm::mat4 &cameraView,
                                                                      const glm::mat4 &cameraProjection) {
    const auto lightDirection = glm::normalize(glm::vec3{0.50f, 1.0f, -0.70f});
    const auto inverseCameraViewProjection = glm::inverse(cameraProjection * cameraView);
    std::array<glm::mat4, cascadeCount> result{};
    float cascadeNearDistance = cameraNearClip;

    for (std::uint32_t cascadeIndex = 0; cascadeIndex < cascadeCount; ++cascadeIndex) {
        const float cascadeFarDistance = cascadeSplitDistances[cascadeIndex];
        const auto projectedNear = cameraProjection * glm::vec4{0.0f, 0.0f, -cascadeNearDistance, 1.0f};
        const auto projectedFar = cameraProjection * glm::vec4{0.0f, 0.0f, -cascadeFarDistance, 1.0f};
        const std::array cascadeDepths = {
            projectedNear.z / projectedNear.w,
            projectedFar.z / projectedFar.w,
        };

        std::array<glm::vec3, 8> corners{};
        std::size_t cornerIndex = 0;
        for (const float cascadeDepth : cascadeDepths) {
            for (const float y : {-1.0f, 1.0f}) {
                for (const float x : {-1.0f, 1.0f}) {
                    const auto worldCorner = inverseCameraViewProjection * glm::vec4{x, y, cascadeDepth, 1.0f};
                    corners[cornerIndex++] = glm::vec3{worldCorner} / worldCorner.w;
                }
            }
        }

        glm::vec3 center{0.0f};
        for (const auto &corner : corners) {
            center += corner;
        }
        center /= static_cast<float>(corners.size());
        const auto lightView =
            glm::lookAtRH(center + lightDirection * (cascadeFarDistance + 50.0f), center, glm::vec3{0.0f, 1.0f, 0.0f});

        glm::vec3 minimum{std::numeric_limits<float>::max()};
        glm::vec3 maximum{std::numeric_limits<float>::lowest()};
        for (const auto &corner : corners) {
            const auto lightSpaceCorner = glm::vec3{lightView * glm::vec4{corner, 1.0f}};
            minimum = glm::min(minimum, lightSpaceCorner);
            maximum = glm::max(maximum, lightSpaceCorner);
        }

        const auto cascadeExtent = maximum - minimum;
        const glm::vec3 padding{
            std::max(4.0f, cascadeExtent.x * 0.12f),
            std::max(4.0f, cascadeExtent.y * 0.12f),
            24.0f,
        };
        minimum -= padding;
        maximum += padding;
        const float nearPlane = std::max(0.1f, -maximum.z);
        const float farPlane = std::max(nearPlane + 0.1f, -minimum.z);
        result[cascadeIndex] =
            glm::orthoRH_ZO(minimum.x, maximum.x, minimum.y, maximum.y, nearPlane, farPlane) * lightView;
        cascadeNearDistance = cascadeFarDistance;
    }
    return result;
}

}  // namespace

class CascadedShadowMappingApplication final : public WindowedVulkanApp {
public:
    CascadedShadowMappingApplication()
        : WindowedVulkanApp({
              .window = {
                  .width = 1024,
                  .height = 768,
                  .title = "Vulkan Cascaded Shadow Maps",
              },
              .apiVersion = vk::ApiVersion11,
          }),
          camera(OrbitCameraConfig{
              .target = {0.0f, 8.0f, 65.0f},
              .yawRadians = -1.57079632679489661923f,
              .pitchRadians = 0.32f,
              .distance = 85.0f,
              .minDistance = 20.0f,
              .maxDistance = 180.0f,
          }) {}

private:
    BufferResource vertexBuffer;
    BufferResource indexBuffer;
    BufferResource cascadeUniformBuffer;
    std::uint32_t indexCount = 0;

    vk::Format shadowMapFormat = vk::Format::eUndefined;
    ShadowMapResource shadowMap;
    ShadowPassResources shadowPass;

    vk::raii::DescriptorSetLayout shadowDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool shadowDescriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> shadowDescriptorSets;

    vk::raii::ShaderModule depthVertexShader = nullptr;
    vk::raii::ShaderModule sceneVertexShader = nullptr;
    vk::raii::ShaderModule sceneFragmentShader = nullptr;
    vk::raii::ShaderModule previewVertexShader = nullptr;
    vk::raii::ShaderModule previewFragmentShader = nullptr;

    vk::raii::PipelineLayout depthPipelineLayout = nullptr;
    vk::raii::PipelineLayout scenePipelineLayout = nullptr;
    vk::raii::PipelineLayout previewPipelineLayout = nullptr;
    vk::raii::Pipeline depthPipeline = nullptr;
    vk::raii::Pipeline scenePipeline = nullptr;
    vk::raii::Pipeline previewPipeline = nullptr;

    OrbitCamera camera;
    std::optional<GlfwCursorPosition> previousDragCursor;
    bool resetKeyWasPressed = false;
    vk::ImageLayout shadowMapLayout = vk::ImageLayout::eUndefined;

    void createApplicationResources() override {
        createGeometryBuffers();
        createCascadeUniformBuffer();
        createShadowMap();
        createShadowRenderPass();
        createShadowDescriptors();
        createPipelines();
    }

    void recordFrameCommands(vk::raii::CommandBuffer &commandBuffer, std::uint32_t imageIndex) override {
        updateInput();

        const auto extent = swapchainExtent();
        const auto projection = glm::perspectiveRH_ZO(
            cameraVerticalFovRadians, static_cast<float>(extent.width) / static_cast<float>(std::max(extent.height, 1u)),
            cameraNearClip, cameraFarClip);
        const auto cameraView = camera.viewMatrix();
        const auto cascadeLightViewProjections = makeCascadeLightViewProjections(cameraView, projection);
        const CascadeUniforms cascadeUniforms{
            .lightViewProjections = cascadeLightViewProjections,
            .cascadeSplits = {cascadeSplitDistances[0], cascadeSplitDistances[1], cascadeSplitDistances[2], 0.0f},
        };
        uploadBuffer(cascadeUniformBuffer, &cascadeUniforms, sizeof(cascadeUniforms));
        const ScenePushConstants scenePushConstants{
            .cameraViewProjection = projection * cameraView,
            .cameraView = cameraView,
        };

        transitionShadowMapToDepthAttachment(commandBuffer);
        for (std::uint32_t cascadeIndex = 0; cascadeIndex < cascadeCount; ++cascadeIndex) {
            const DepthPushConstants depthPushConstants{
                .lightViewProjection = cascadeLightViewProjections[cascadeIndex],
            };
            beginShadowRenderPass(commandBuffer, cascadeIndex);
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *depthPipeline);
            bindSceneGeometry(commandBuffer);
            commandBuffer.pushConstants(*depthPipelineLayout, vk::ShaderStageFlagBits::eVertex, 0,
                                        static_cast<std::uint32_t>(sizeof(depthPushConstants)), &depthPushConstants);
            commandBuffer.drawIndexed(indexCount, 1, 0, 0, 0);
            commandBuffer.endRenderPass();
        }
        transitionShadowMapToShaderRead(commandBuffer);

        beginSwapchainRenderPass(commandBuffer, imageIndex);
        const std::array descriptorSetHandles = {*shadowDescriptorSets.front()};
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
        const auto scene = makeLargeSceneGeometry();
        if (scene.vertices.empty() || scene.indices.empty()) {
            throw std::runtime_error("CSM outdoor scene must contain geometry");
        }
        indexCount = static_cast<std::uint32_t>(scene.indices.size());
        const auto vertexBufferSize = static_cast<vk::DeviceSize>(scene.vertices.size() * sizeof(Vertex));
        vertexBuffer = createHostVisibleBuffer(vertexBufferSize, vk::BufferUsageFlagBits::eVertexBuffer);
        uploadBuffer(vertexBuffer, scene.vertices.data(), vertexBufferSize);

        const auto indexBufferSize = static_cast<vk::DeviceSize>(scene.indices.size() * sizeof(std::uint32_t));
        indexBuffer = createHostVisibleBuffer(indexBufferSize, vk::BufferUsageFlagBits::eIndexBuffer);
        uploadBuffer(indexBuffer, scene.indices.data(), indexBufferSize);
    }

    void createCascadeUniformBuffer() {
        cascadeUniformBuffer = createHostVisibleBuffer(sizeof(CascadeUniforms), vk::BufferUsageFlagBits::eUniformBuffer);
    }

    [[nodiscard]] vk::Format findSampledDepthFormat() const {
        constexpr std::array candidates = {
            vk::Format::eD32Sfloat,
            vk::Format::eD16Unorm,
        };
        constexpr vk::FormatFeatureFlags requiredFeatures =
            vk::FormatFeatureFlagBits::eDepthStencilAttachment | vk::FormatFeatureFlagBits::eSampledImage;
        for (const auto format : candidates) {
            const auto features = physicalDevice().getFormatProperties(format).optimalTilingFeatures;
            if ((features & requiredFeatures) == requiredFeatures) {
                return format;
            }
        }
        throw std::runtime_error("No depth-only format supports both shadow rendering and sampled reads");
    }

    void createShadowMap() {
        shadowMapFormat = findSampledDepthFormat();
        shadowMap.image = device().createImage({
            .imageType = vk::ImageType::e2D,
            .format = shadowMapFormat,
            .extent = {shadowMapSize, shadowMapSize, 1},
            .mipLevels = 1,
            .arrayLayers = cascadeCount,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
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

        shadowMap.arrayImageView = device().createImageView({
            .image = *shadowMap.image,
            .viewType = vk::ImageViewType::e2DArray,
            .format = shadowMapFormat,
            .subresourceRange =
                {
                    .aspectMask = vk::ImageAspectFlagBits::eDepth,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = cascadeCount,
                },
        });
        shadowMap.layerImageViews.reserve(cascadeCount);
        for (std::uint32_t cascadeIndex = 0; cascadeIndex < cascadeCount; ++cascadeIndex) {
            shadowMap.layerImageViews.emplace_back(device().createImageView({
                .image = *shadowMap.image,
                .viewType = vk::ImageViewType::e2D,
                .format = shadowMapFormat,
                .subresourceRange =
                    {
                        .aspectMask = vk::ImageAspectFlagBits::eDepth,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = cascadeIndex,
                        .layerCount = 1,
                    },
            }));
        }

        shadowMap.comparisonSampler = device().createSampler({
            .magFilter = vk::Filter::eNearest,
            .minFilter = vk::Filter::eNearest,
            .mipmapMode = vk::SamplerMipmapMode::eNearest,
            .addressModeU = vk::SamplerAddressMode::eClampToBorder,
            .addressModeV = vk::SamplerAddressMode::eClampToBorder,
            .addressModeW = vk::SamplerAddressMode::eClampToBorder,
            .compareEnable = true,
            .compareOp = vk::CompareOp::eLessOrEqual,
            .minLod = 0.0f,
            .maxLod = 0.0f,
            .borderColor = vk::BorderColor::eFloatOpaqueWhite,
        });
        shadowMap.previewSampler = device().createSampler({
            .magFilter = vk::Filter::eNearest,
            .minFilter = vk::Filter::eNearest,
            .mipmapMode = vk::SamplerMipmapMode::eNearest,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
            .minLod = 0.0f,
            .maxLod = 0.0f,
        });
    }

    void createShadowRenderPass() {
        const vk::AttachmentDescription depthAttachment{
            .format = shadowMapFormat,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        };
        const vk::AttachmentReference depthReference{
            .attachment = 0,
            .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        };
        const vk::SubpassDescription subpass{
            .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
            .pDepthStencilAttachment = &depthReference,
        };
        shadowPass.renderPass = device().createRenderPass({
            .attachmentCount = 1,
            .pAttachments = &depthAttachment,
            .subpassCount = 1,
            .pSubpasses = &subpass,
        });

        shadowPass.framebuffers.reserve(cascadeCount);
        for (std::uint32_t cascadeIndex = 0; cascadeIndex < cascadeCount; ++cascadeIndex) {
            const vk::ImageView attachment = *shadowMap.layerImageViews[cascadeIndex];
            shadowPass.framebuffers.emplace_back(device().createFramebuffer({
                .renderPass = *shadowPass.renderPass,
                .attachmentCount = 1,
                .pAttachments = &attachment,
                .width = shadowMapSize,
                .height = shadowMapSize,
                .layers = 1,
            }));
        }
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
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 2,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
            },
        };
        shadowDescriptorSetLayout = device().createDescriptorSetLayout({
            .bindingCount = static_cast<std::uint32_t>(bindings.size()),
            .pBindings = bindings.data(),
        });

        const std::array poolSizes = {
            vk::DescriptorPoolSize{
                .type = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = 2,
            },
            vk::DescriptorPoolSize{
                .type = vk::DescriptorType::eUniformBuffer,
                .descriptorCount = 1,
            },
        };
        shadowDescriptorPool = device().createDescriptorPool({
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = 1,
            .poolSizeCount = static_cast<std::uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data(),
        });

        const vk::DescriptorSetLayout layout = *shadowDescriptorSetLayout;
        shadowDescriptorSets = device().allocateDescriptorSets({
            .descriptorPool = *shadowDescriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &layout,
        });

        const vk::DescriptorImageInfo comparisonImageInfo{
            .sampler = *shadowMap.comparisonSampler,
            .imageView = *shadowMap.arrayImageView,
            .imageLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal,
        };
        const vk::DescriptorBufferInfo cascadeUniformInfo{
            .buffer = *cascadeUniformBuffer.buffer,
            .offset = 0,
            .range = sizeof(CascadeUniforms),
        };
        const vk::DescriptorImageInfo previewImageInfo{
            .sampler = *shadowMap.previewSampler,
            .imageView = *shadowMap.arrayImageView,
            .imageLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal,
        };
        const std::array writes = {
            vk::WriteDescriptorSet{
                .dstSet = *shadowDescriptorSets.front(),
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &comparisonImageInfo,
            },
            vk::WriteDescriptorSet{
                .dstSet = *shadowDescriptorSets.front(),
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo = &cascadeUniformInfo,
            },
            vk::WriteDescriptorSet{
                .dstSet = *shadowDescriptorSets.front(),
                .dstBinding = 2,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &previewImageInfo,
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
        sceneVertexShader = loadShaderModule(shaderDirectory / "shadow_scene.vert.spv");
        sceneFragmentShader = loadShaderModule(shaderDirectory / "shadow_scene.frag.spv");
        previewVertexShader = loadShaderModule(shaderDirectory / "shadow_preview.vert.spv");
        previewFragmentShader = loadShaderModule(shaderDirectory / "shadow_preview.frag.spv");

        createDepthPipeline();
        createScenePipeline();
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
        const vk::PipelineColorBlendStateCreateInfo colorBlending{};
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

    void beginShadowRenderPass(vk::raii::CommandBuffer &commandBuffer, std::uint32_t cascadeIndex) const {
        vk::ClearValue clearValue{};
        clearValue.depthStencil = vk::ClearDepthStencilValue{1.0f, 0};
        const vk::RenderPassBeginInfo beginInfo{
            .renderPass = *shadowPass.renderPass,
            .framebuffer = *shadowPass.framebuffers[cascadeIndex],
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
        const auto oldLayout = shadowMapLayout;
        const auto sourceStage = oldLayout == vk::ImageLayout::eUndefined
                                     ? vk::PipelineStageFlags{vk::PipelineStageFlagBits::eTopOfPipe}
                                     : vk::PipelineStageFlags{vk::PipelineStageFlagBits::eFragmentShader};
        const auto sourceAccess = oldLayout == vk::ImageLayout::eUndefined
                                      ? vk::AccessFlags{}
                                      : vk::AccessFlags{vk::AccessFlagBits::eShaderRead};
        transitionShadowMap(commandBuffer, oldLayout, vk::ImageLayout::eDepthStencilAttachmentOptimal, sourceStage,
                            vk::PipelineStageFlagBits::eEarlyFragmentTests |
                                vk::PipelineStageFlagBits::eLateFragmentTests,
                            sourceAccess, vk::AccessFlagBits::eDepthStencilAttachmentRead |
                                              vk::AccessFlagBits::eDepthStencilAttachmentWrite);
    }

    void transitionShadowMapToShaderRead(vk::raii::CommandBuffer &commandBuffer) {
        transitionShadowMap(commandBuffer, vk::ImageLayout::eDepthStencilAttachmentOptimal,
                            vk::ImageLayout::eDepthStencilReadOnlyOptimal,
                            vk::PipelineStageFlagBits::eEarlyFragmentTests |
                                vk::PipelineStageFlagBits::eLateFragmentTests,
                            vk::PipelineStageFlagBits::eFragmentShader,
                            vk::AccessFlagBits::eDepthStencilAttachmentWrite, vk::AccessFlagBits::eShaderRead);
    }

    void transitionShadowMap(vk::raii::CommandBuffer &commandBuffer,
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
            .image = *shadowMap.image,
            .subresourceRange =
                {
                    .aspectMask = vk::ImageAspectFlagBits::eDepth,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = cascadeCount,
                },
        };
        commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, {barrier});
        shadowMapLayout = newLayout;
    }
};

int main() {
    try {
        CascadedShadowMappingApplication application;
        application.run();
    } catch (const std::exception &error) {
        std::println("Error: {}", error.what());
        return 1;
    }
    return 0;
}
