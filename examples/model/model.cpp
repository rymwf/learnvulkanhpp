#include <glm/ext/matrix_clip_space.hpp>
#include <glm/glm.hpp>
#include <tiny_obj_loader.h>

import vulkan;
import std;
import common;

#ifndef BUNNY_OBJ_PATH
#error "BUNNY_OBJ_PATH must be set by CMake (see examples/model/CMakeLists.txt)"
#endif

#ifndef SHADER_SPV_DIR
#error "SHADER_SPV_DIR must be set by CMake (see examples/model/CMakeLists.txt)"
#endif

namespace {

struct Vertex {
    std::array<float, 3> position;
    std::array<float, 3> normal;
};

static_assert(sizeof(Vertex) == 6 * sizeof(float));

struct BufferResource {
    vk::raii::DeviceMemory memory = nullptr;
    vk::raii::Buffer buffer = nullptr;
};

struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
};

struct ViewProjectionPushConstants {
    glm::mat4 viewProjection{1.0f};
};

static_assert(sizeof(ViewProjectionPushConstants) == 16 * sizeof(float));

constexpr float minimumNormalLengthSquared = 1.0e-20f;
constexpr float cameraVerticalFovRadians = 0.78539816339744830962f;
constexpr float orbitRadiansPerPixel = 0.005f;

glm::vec3 readPosition(const tinyobj::attrib_t &attributes, std::uint32_t index) {
    const auto offset = static_cast<std::size_t>(index) * 3;
    return {
        static_cast<float>(attributes.vertices[offset]),
        static_cast<float>(attributes.vertices[offset + 1]),
        static_cast<float>(attributes.vertices[offset + 2]),
    };
}

MeshData loadBunnyMesh() {
    tinyobj::ObjReaderConfig readerConfig;
    readerConfig.triangulate = true;
    readerConfig.vertex_color = false;

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(BUNNY_OBJ_PATH, readerConfig)) {
        const auto &error = reader.Error();
        throw std::runtime_error(error.empty() ? std::format("Failed to load OBJ: {}", BUNNY_OBJ_PATH)
                                               : std::format("Failed to load OBJ {}: {}", BUNNY_OBJ_PATH, error));
    }
    if (!reader.Warning().empty()) {
        std::println("OBJ warning: {}", reader.Warning());
    }

    const auto &attributes = reader.GetAttrib();
    if (attributes.vertices.empty() || attributes.vertices.size() % 3 != 0) {
        throw std::runtime_error("OBJ has no valid position data");
    }

    const auto vertexCount = attributes.vertices.size() / 3;
    if (vertexCount > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("OBJ has too many vertices for uint32 indices");
    }

    std::vector<glm::vec3> positions;
    positions.reserve(vertexCount);
    glm::vec3 minimum{std::numeric_limits<float>::max()};
    glm::vec3 maximum{std::numeric_limits<float>::lowest()};
    for (std::uint32_t index = 0; index < vertexCount; ++index) {
        const auto position = readPosition(attributes, index);
        positions.push_back(position);
        minimum = glm::min(minimum, position);
        maximum = glm::max(maximum, position);
    }

    std::vector<glm::vec3> normalSums(vertexCount, glm::vec3{0.0f});
    std::vector<std::uint32_t> indices;
    for (const auto &shape : reader.GetShapes()) {
        std::size_t indexOffset = 0;
        for (const auto faceVertexCount : shape.mesh.num_face_vertices) {
            if (faceVertexCount != 3) {
                throw std::runtime_error("OBJ triangulation produced a non-triangle face");
            }
            if (indexOffset + faceVertexCount > shape.mesh.indices.size()) {
                throw std::runtime_error("OBJ shape has incomplete face indices");
            }

            std::array<std::uint32_t, 3> triangle{};
            for (std::size_t vertex = 0; vertex < triangle.size(); ++vertex) {
                const auto sourceIndex = shape.mesh.indices[indexOffset + vertex].vertex_index;
                if (sourceIndex < 0 || static_cast<std::size_t>(sourceIndex) >= vertexCount) {
                    throw std::runtime_error("OBJ contains an invalid position index");
                }
                triangle[vertex] = static_cast<std::uint32_t>(sourceIndex);
            }

            const auto faceNormal = glm::cross(positions[triangle[1]] - positions[triangle[0]],
                                               positions[triangle[2]] - positions[triangle[0]]);
            if (glm::dot(faceNormal, faceNormal) > minimumNormalLengthSquared) {
                for (const auto vertex : triangle) {
                    normalSums[vertex] += faceNormal;
                }
            }
            indices.insert(indices.end(), triangle.begin(), triangle.end());
            indexOffset += faceVertexCount;
        }
        if (indexOffset != shape.mesh.indices.size()) {
            throw std::runtime_error("OBJ shape index count does not match its faces");
        }
    }

    if (indices.empty()) {
        throw std::runtime_error("OBJ contains no triangles");
    }
    if (indices.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("OBJ has too many indices for Vulkan drawIndexed");
    }

    const auto extent = maximum - minimum;
    const auto largestExtent = std::max({extent.x, extent.y, extent.z});
    if (largestExtent <= std::numeric_limits<float>::epsilon()) {
        throw std::runtime_error("OBJ has a degenerate bounding box");
    }

    const auto center = (minimum + maximum) * 0.5f;
    const auto scale = 1.8f / largestExtent;
    std::vector<Vertex> vertices;
    vertices.reserve(vertexCount);
    for (std::size_t index = 0; index < vertexCount; ++index) {
        const auto position = (positions[index] - center) * scale;
        const auto squaredNormalLength = glm::dot(normalSums[index], normalSums[index]);
        const auto normal = squaredNormalLength > minimumNormalLengthSquared
                                ? glm::normalize(normalSums[index])
                                : glm::vec3{0.0f, 1.0f, 0.0f};
        vertices.push_back({
            .position = {position.x, position.y, position.z},
            .normal = {normal.x, normal.y, normal.z},
        });
    }

    std::println("Loaded bunny.obj: {} vertices, {} indices", vertices.size(), indices.size());
    return {.vertices = std::move(vertices), .indices = std::move(indices)};
}

}  // namespace

class TinyObjBunnyApplication : public WindowedVulkanApp {
public:
    TinyObjBunnyApplication()
        : WindowedVulkanApp({
              .window = {
                  .width = 1024,
                  .height = 768,
                  .title = "Vulkan TinyObjLoader Bunny",
              },
              .apiVersion = vk::ApiVersion10,
          }) {}

private:
    std::vector<vk::raii::ImageView> imageViews;
    std::vector<vk::raii::Image> depthImages;
    std::vector<vk::raii::DeviceMemory> depthImageMemories;
    std::vector<vk::raii::ImageView> depthImageViews;

    BufferResource vertexBuffer;
    BufferResource indexBuffer;
    std::uint32_t indexCount = 0;

    vk::raii::RenderPass renderPass = nullptr;
    vk::raii::ShaderModule vertShaderModule = nullptr;
    vk::raii::ShaderModule fragShaderModule = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;
    std::vector<vk::raii::Framebuffer> framebuffers;
    OrbitCamera camera;
    std::optional<GlfwCursorPosition> previousDragCursor;
    bool resetKeyWasPressed = false;

    void createApplicationResources() override {
        createGeometryBuffers();
        createRenderTargets();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
    }

    void recordFrameCommands(vk::raii::CommandBuffer &commandBuffer, std::uint32_t imageIndex) override {
        std::array<vk::ClearValue, 2> clearValues{};
        clearValues[0].color.setFloat32(std::array<float, 4>{0.02f, 0.02f, 0.08f, 1.0f});
        clearValues[1].depthStencil = vk::ClearDepthStencilValue{1.0f, 0};

        vk::RenderPassBeginInfo renderPassInfo{
            .renderPass = *renderPass,
            .framebuffer = *framebuffers[imageIndex],
            .renderArea = {.offset = {0, 0}, .extent = swapchainExtent()},
            .clearValueCount = static_cast<std::uint32_t>(clearValues.size()),
            .pClearValues = clearValues.data(),
        };

        updateCameraFromInput();

        const auto extent = swapchainExtent();
        auto projection = glm::perspectiveRH_ZO(cameraVerticalFovRadians,
                                                static_cast<float>(extent.width) / static_cast<float>(extent.height),
                                                0.1f, 10.0f);
        projection[1][1] *= -1.0f;
        const ViewProjectionPushConstants pushConstants{
            .viewProjection = projection * camera.viewMatrix(),
        };

        const vk::Buffer rawVertexBuffer = *vertexBuffer.buffer;
        constexpr vk::DeviceSize vertexOffset = 0;
        commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
        commandBuffer.bindVertexBuffers(0, {rawVertexBuffer}, {vertexOffset});
        commandBuffer.bindIndexBuffer(*indexBuffer.buffer, 0, vk::IndexType::eUint32);
        commandBuffer.pushConstants(*pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0,
                                    static_cast<std::uint32_t>(sizeof(pushConstants)), &pushConstants);
        commandBuffer.drawIndexed(indexCount, 1, 0, 0, 0);
        commandBuffer.endRenderPass();
    }

    void updateCameraFromInput() {
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
                    camera.orbit(-horizontalPixels * orbitRadiansPerPixel,
                                 -verticalPixels * orbitRadiansPerPixel);
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

    BufferResource createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage) {
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
        const auto mesh = loadBunnyMesh();
        indexCount = static_cast<std::uint32_t>(mesh.indices.size());

        vertexBuffer = createBuffer(mesh.vertices.size() * sizeof(Vertex), vk::BufferUsageFlagBits::eVertexBuffer);
        uploadBuffer(vertexBuffer, mesh.vertices.data(), mesh.vertices.size() * sizeof(Vertex));

        indexBuffer = createBuffer(mesh.indices.size() * sizeof(std::uint32_t), vk::BufferUsageFlagBits::eIndexBuffer);
        uploadBuffer(indexBuffer, mesh.indices.data(), mesh.indices.size() * sizeof(std::uint32_t));
    }

    void createRenderTargets() {
        const auto extent = swapchainExtent();
        const auto surfaceFormat = swapchainSurfaceFormat();
        for (const auto image : swapchainImages()) {
            imageViews.push_back(device().createImageView({
                .image = image,
                .viewType = vk::ImageViewType::e2D,
                .format = surfaceFormat.format,
                .subresourceRange =
                    {
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
            }));

            depthImages.push_back(device().createImage({
                .imageType = vk::ImageType::e2D,
                .format = vk::Format::eD32Sfloat,
                .extent = {extent.width, extent.height, 1},
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = vk::SampleCountFlagBits::e1,
                .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
            }));
            const auto memoryRequirements = depthImages.back().getMemoryRequirements();
            depthImageMemories.emplace_back(
                device(),
                vk::MemoryAllocateInfo{
                    .allocationSize = memoryRequirements.size,
                    .memoryTypeIndex = findMemoryTypeIndex(memoryRequirements,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal),
                });
            depthImages.back().bindMemory(*depthImageMemories.back(), 0);

            depthImageViews.push_back(device().createImageView({
                .image = *depthImages.back(),
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
            }));
        }
    }

    void createRenderPass() {
        const auto surfaceFormat = swapchainSurfaceFormat();
        std::array<vk::AttachmentDescription, 2> attachments = {
            vk::AttachmentDescription{
                .format = surfaceFormat.format,
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
        const vk::SubpassDependency dependency{
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
        renderPass = device().createRenderPass({
            .attachmentCount = static_cast<std::uint32_t>(attachments.size()),
            .pAttachments = attachments.data(),
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 1,
            .pDependencies = &dependency,
        });
    }

    void createGraphicsPipeline() {
        const std::filesystem::path shaderDir{SHADER_SPV_DIR};
        const auto vertSpirv = readSpirvFile(shaderDir / "model.vert.spv");
        const auto fragSpirv = readSpirvFile(shaderDir / "model.frag.spv");
        vertShaderModule = device().createShaderModule({
            .codeSize = vertSpirv.size() * sizeof(std::uint32_t),
            .pCode = vertSpirv.data(),
        });
        fragShaderModule = device().createShaderModule({
            .codeSize = fragSpirv.size() * sizeof(std::uint32_t),
            .pCode = fragSpirv.data(),
        });

        std::array shaderStages = {
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = *vertShaderModule,
                .pName = "main",
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = *fragShaderModule,
                .pName = "main",
            },
        };
        const vk::VertexInputBindingDescription bindingDescription{
            .binding = 0,
            .stride = static_cast<std::uint32_t>(sizeof(Vertex)),
            .inputRate = vk::VertexInputRate::eVertex,
        };
        std::array attributeDescriptions = {
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
        };
        const vk::PipelineVertexInputStateCreateInfo vertexInput{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions = attributeDescriptions.data(),
        };
        const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::eTriangleList,
        };

        const auto extent = swapchainExtent();
        const vk::Viewport viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(extent.width),
            .height = static_cast<float>(extent.height),
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
            .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eClockwise,
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
            .size = static_cast<std::uint32_t>(sizeof(ViewProjectionPushConstants)),
        };
        pipelineLayout = device().createPipelineLayout({
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstantRange,
        });
        graphicsPipeline = device().createGraphicsPipeline(nullptr, {
                                                                      .stageCount = static_cast<std::uint32_t>(shaderStages.size()),
                                                                      .pStages = shaderStages.data(),
                                                                      .pVertexInputState = &vertexInput,
                                                                      .pInputAssemblyState = &inputAssembly,
                                                                      .pViewportState = &viewportState,
                                                                      .pRasterizationState = &rasterizer,
                                                                      .pMultisampleState = &multisampling,
                                                                      .pDepthStencilState = &depthStencil,
                                                                      .pColorBlendState = &colorBlending,
                                                                      .layout = *pipelineLayout,
                                                                      .renderPass = *renderPass,
                                                                      .subpass = 0,
                                                                  });
    }

    void createFramebuffers() {
        const auto extent = swapchainExtent();
        for (std::size_t index = 0; index < swapchainImages().size(); ++index) {
            std::array<vk::ImageView, 2> attachments = {*imageViews[index], *depthImageViews[index]};
            framebuffers.push_back(device().createFramebuffer({
                .renderPass = *renderPass,
                .attachmentCount = static_cast<std::uint32_t>(attachments.size()),
                .pAttachments = attachments.data(),
                .width = extent.width,
                .height = extent.height,
                .layers = 1,
            }));
        }
    }
};

int main() {
    try {
        TinyObjBunnyApplication app;
        app.run();
    } catch (const std::exception &error) {
        std::println("Error: {}", error.what());
        return 1;
    }
    return 0;
}
