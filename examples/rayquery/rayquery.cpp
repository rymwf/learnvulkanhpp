#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

import vulkan;
import std;
import common;

#ifndef SHADER_SPV_DIR
#error "SHADER_SPV_DIR must be set by CMake (see examples/rayquery/CMakeLists.txt)"
#endif

namespace {

struct BufferResource {
    vk::raii::DeviceMemory memory = nullptr;
    vk::raii::Buffer buffer = nullptr;
    vk::DeviceSize size = 0;
};

struct AccelerationStructureResource {
    BufferResource storage;
    vk::raii::AccelerationStructureKHR handle = nullptr;
};

struct Vertex {
    std::array<float, 3> position;
    std::array<float, 3> normal;
    std::array<float, 3> color;
};

static_assert(sizeof(Vertex) == 9 * sizeof(float));

struct MeshResource {
    BufferResource vertexBuffer;
    BufferResource indexBuffer;
    std::uint32_t vertexCount = 0;
    std::uint32_t indexCount = 0;
    AccelerationStructureResource bottomLevelAS;
};

struct ViewProjectionPushConstants {
    glm::mat4 viewProjection{1.0f};
};

static_assert(sizeof(ViewProjectionPushConstants) == 16 * sizeof(float));

template <typename T>
constexpr T alignUp(T value, T alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

}  // namespace

class RayQueryApplication : public WindowedVulkanApp {
public:
    RayQueryApplication()
        : WindowedVulkanApp({
              .window = {
                  .width = 1024,
                  .height = 768,
                  .title = "Vulkan Ray Query Shadows",
              },
              .apiVersion = vk::ApiVersion12,
              .requiredDeviceExtensions = {
                  "VK_KHR_swapchain",
                  "VK_KHR_deferred_host_operations",
                  "VK_KHR_acceleration_structure",
                  "VK_KHR_ray_query",
              },
          }) {}

private:
    vk::PhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructureProperties{};

    MeshResource ground;
    MeshResource occluder;
    BufferResource instanceBuffer;
    AccelerationStructureResource topLevelAS;

    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> descriptorSets;

    std::vector<vk::raii::ShaderModule> shaderModules;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;

    void createApplicationResources() override {
        loadAccelerationStructureProperties();
        createGeometryBuffers();
        createAccelerationStructures();
        createDescriptors();
        createGraphicsPipeline();
    }

    void releaseSwapchainDependentResources() override {
        graphicsPipeline = nullptr;
        pipelineLayout = nullptr;
    }

    void recreateSwapchainDependentResources() override {
        createGraphicsPipeline();
    }

    DeviceFeatureChain enabledDeviceFeatureChain() const override {
        return DeviceFeatureChain::make(
            vk::PhysicalDeviceFeatures{},
            vk::PhysicalDeviceBufferDeviceAddressFeatures{
                .bufferDeviceAddress = true,
            },
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR{
                .accelerationStructure = true,
            },
            vk::PhysicalDeviceRayQueryFeaturesKHR{
                .rayQuery = true,
            });
    }

    void loadAccelerationStructureProperties() {
        const auto properties = physicalDevice().getProperties2<vk::PhysicalDeviceProperties2,
                                                                 vk::PhysicalDeviceAccelerationStructurePropertiesKHR>();
        accelerationStructureProperties = properties.get<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>();
    }

    BufferResource createBuffer(vk::DeviceSize size,
                                vk::BufferUsageFlags usage,
                                vk::MemoryPropertyFlags memoryProperties,
                                bool needsDeviceAddress) {
        BufferResource resource;
        resource.size = size;
        resource.buffer = device().createBuffer({
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
        });

        const auto requirements = resource.buffer.getMemoryRequirements();
        vk::MemoryAllocateFlagsInfo allocateFlagsInfo{
            .flags = needsDeviceAddress ? vk::MemoryAllocateFlagBits::eDeviceAddress : vk::MemoryAllocateFlags{},
        };
        vk::MemoryAllocateInfo allocateInfo{
            .pNext = needsDeviceAddress ? &allocateFlagsInfo : nullptr,
            .allocationSize = requirements.size,
            .memoryTypeIndex = findMemoryTypeIndex(requirements, memoryProperties),
        };
        resource.memory = vk::raii::DeviceMemory(device(), allocateInfo);
        resource.buffer.bindMemory(*resource.memory, 0);
        return resource;
    }

    void uploadBuffer(const BufferResource &resource, const void *data, vk::DeviceSize size) {
        void *mapped = resource.memory.mapMemory(0, size);
        std::memcpy(mapped, data, static_cast<std::size_t>(size));
        resource.memory.unmapMemory();
    }

    vk::DeviceAddress getBufferAddress(const BufferResource &resource) const {
        return device().getBufferAddress({.buffer = *resource.buffer});
    }

    template <std::size_t VertexCount, std::size_t IndexCount>
    MeshResource createMesh(const std::array<Vertex, VertexCount> &vertices,
                            const std::array<std::uint32_t, IndexCount> &indices) {
        static_assert(VertexCount > 0);
        static_assert(IndexCount > 0 && IndexCount % 3 == 0);

        constexpr auto hostMemory =
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
        const auto accelerationStructureInputUsage =
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress;

        MeshResource mesh;
        mesh.vertexCount = static_cast<std::uint32_t>(VertexCount);
        mesh.indexCount = static_cast<std::uint32_t>(IndexCount);
        mesh.vertexBuffer = createBuffer(sizeof(vertices),
                                         accelerationStructureInputUsage | vk::BufferUsageFlagBits::eVertexBuffer,
                                         hostMemory, true);
        uploadBuffer(mesh.vertexBuffer, vertices.data(), sizeof(vertices));
        mesh.indexBuffer = createBuffer(sizeof(indices),
                                        accelerationStructureInputUsage | vk::BufferUsageFlagBits::eIndexBuffer,
                                        hostMemory, true);
        uploadBuffer(mesh.indexBuffer, indices.data(), sizeof(indices));
        return mesh;
    }

    void createGeometryBuffers() {
        constexpr std::array<Vertex, 4> groundVertices = {
            Vertex{{-2.2f, 0.0f, -2.0f}, {0.0f, 1.0f, 0.0f}, {0.16f, 0.48f, 0.58f}},
            Vertex{{2.2f, 0.0f, -2.0f}, {0.0f, 1.0f, 0.0f}, {0.16f, 0.48f, 0.58f}},
            Vertex{{2.2f, 0.0f, 2.0f}, {0.0f, 1.0f, 0.0f}, {0.16f, 0.48f, 0.58f}},
            Vertex{{-2.2f, 0.0f, 2.0f}, {0.0f, 1.0f, 0.0f}, {0.16f, 0.48f, 0.58f}},
        };
        constexpr std::array<std::uint32_t, 6> groundIndices = {0, 2, 1, 0, 3, 2};
        constexpr std::array<Vertex, 3> occluderVertices = {
            Vertex{{-0.65f, 0.72f, -0.45f}, {0.0f, 1.0f, 0.0f}, {0.92f, 0.34f, 0.14f}},
            Vertex{{0.70f, 0.72f, -0.28f}, {0.0f, 1.0f, 0.0f}, {0.92f, 0.34f, 0.14f}},
            Vertex{{0.08f, 0.72f, 0.78f}, {0.0f, 1.0f, 0.0f}, {0.92f, 0.34f, 0.14f}},
        };
        constexpr std::array<std::uint32_t, 3> occluderIndices = {0, 2, 1};

        ground = createMesh(groundVertices, groundIndices);
        occluder = createMesh(occluderVertices, occluderIndices);
    }

    template <typename Recorder>
    void submitImmediate(Recorder &&recorder) {
        auto buffers = device().allocateCommandBuffers({
            .commandPool = *commandPool(),
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        });
        auto &commandBuffer = buffers.front();
        commandBuffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        std::invoke(std::forward<Recorder>(recorder), commandBuffer);
        commandBuffer.end();

        const vk::CommandBuffer rawCommandBuffer = *commandBuffer;
        const vk::SubmitInfo submitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &rawCommandBuffer,
        };
        graphicsQueue().submit(submitInfo);
        graphicsQueue().waitIdle();
    }

    AccelerationStructureResource createAccelerationStructure(vk::AccelerationStructureTypeKHR type,
                                                              vk::DeviceSize size) {
        AccelerationStructureResource resource;
        resource.storage = createBuffer(size,
                                        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                                            vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                        vk::MemoryPropertyFlagBits::eDeviceLocal, true);
        resource.handle = vk::raii::AccelerationStructureKHR(device(), {
                                                                         .buffer = *resource.storage.buffer,
                                                                         .offset = 0,
                                                                         .size = size,
                                                                         .type = type,
                                                                     });
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

    void synchronizeAccelerationStructureBuild(vk::PipelineStageFlags destinationStage) {
        submitImmediate([destinationStage](vk::raii::CommandBuffer &commandBuffer) {
            const vk::MemoryBarrier barrier{
                .srcAccessMask = vk::AccessFlagBits::eAccelerationStructureWriteKHR,
                .dstAccessMask = vk::AccessFlagBits::eAccelerationStructureReadKHR,
            };
            commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, destinationStage,
                                          {}, {barrier}, {}, {});
        });
    }

    void buildBottomLevelAccelerationStructure(MeshResource &mesh) {
        vk::AccelerationStructureGeometryTrianglesDataKHR triangles{
            .vertexFormat = vk::Format::eR32G32B32Sfloat,
            .vertexData = {.deviceAddress = getBufferAddress(mesh.vertexBuffer)},
            .vertexStride = sizeof(Vertex),
            .maxVertex = mesh.vertexCount - 1,
            .indexType = vk::IndexType::eUint32,
            .indexData = {.deviceAddress = getBufferAddress(mesh.indexBuffer)},
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
        const std::array primitiveCounts = {mesh.indexCount / 3};
        const auto sizes = device().getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo, primitiveCounts);

        mesh.bottomLevelAS =
            createAccelerationStructure(vk::AccelerationStructureTypeKHR::eBottomLevel, sizes.accelerationStructureSize);
        auto scratch = createScratchBuffer(sizes.buildScratchSize);
        buildInfo.dstAccelerationStructure = *mesh.bottomLevelAS.handle;
        buildInfo.scratchData.deviceAddress = alignedScratchAddress(scratch);
        const vk::AccelerationStructureBuildRangeInfoKHR rangeInfo{
            .primitiveCount = primitiveCounts.front(),
        };
        const std::array buildInfos = {buildInfo};
        const std::array<const vk::AccelerationStructureBuildRangeInfoKHR *, 1> rangeInfos = {&rangeInfo};
        submitImmediate([&](vk::raii::CommandBuffer &commandBuffer) {
            commandBuffer.buildAccelerationStructuresKHR(buildInfos, rangeInfos);
        });
    }

    vk::AccelerationStructureInstanceKHR makeInstance(const AccelerationStructureResource &bottomLevelAS,
                                                       std::uint32_t customIndex) const {
        const vk::AccelerationStructureDeviceAddressInfoKHR addressInfo{
            .accelerationStructure = *bottomLevelAS.handle,
        };
        vk::AccelerationStructureInstanceKHR instance{};
        instance.transform.matrix[0][0] = 1.0f;
        instance.transform.matrix[1][1] = 1.0f;
        instance.transform.matrix[2][2] = 1.0f;
        instance.instanceCustomIndex = customIndex;
        instance.mask = 0xff;
        instance.instanceShaderBindingTableRecordOffset = 0;
        instance.flags = static_cast<std::uint32_t>(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);
        instance.accelerationStructureReference = device().getAccelerationStructureAddressKHR(addressInfo);
        return instance;
    }

    void createTopLevelAccelerationStructure() {
        const std::array instances = {
            makeInstance(ground.bottomLevelAS, 0),
            makeInstance(occluder.bottomLevelAS, 1),
        };
        constexpr auto hostMemory =
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
        instanceBuffer = createBuffer(sizeof(instances),
                                      vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
                                          vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                      hostMemory, true);
        uploadBuffer(instanceBuffer, instances.data(), sizeof(instances));

        vk::AccelerationStructureGeometryInstancesDataKHR instanceData{
            .arrayOfPointers = false,
            .data = {.deviceAddress = getBufferAddress(instanceBuffer)},
        };
        vk::AccelerationStructureGeometryKHR geometry{
            .geometryType = vk::GeometryTypeKHR::eInstances,
            .geometry = {.instances = instanceData},
        };
        vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{
            .type = vk::AccelerationStructureTypeKHR::eTopLevel,
            .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
            .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
            .geometryCount = 1,
            .pGeometries = &geometry,
        };
        const std::array primitiveCounts = {static_cast<std::uint32_t>(instances.size())};
        const auto sizes = device().getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo, primitiveCounts);

        topLevelAS =
            createAccelerationStructure(vk::AccelerationStructureTypeKHR::eTopLevel, sizes.accelerationStructureSize);
        auto scratch = createScratchBuffer(sizes.buildScratchSize);
        buildInfo.dstAccelerationStructure = *topLevelAS.handle;
        buildInfo.scratchData.deviceAddress = alignedScratchAddress(scratch);
        const vk::AccelerationStructureBuildRangeInfoKHR rangeInfo{
            .primitiveCount = primitiveCounts.front(),
        };
        const std::array buildInfos = {buildInfo};
        const std::array<const vk::AccelerationStructureBuildRangeInfoKHR *, 1> rangeInfos = {&rangeInfo};
        submitImmediate([&](vk::raii::CommandBuffer &commandBuffer) {
            commandBuffer.buildAccelerationStructuresKHR(buildInfos, rangeInfos);
        });
    }

    void createAccelerationStructures() {
        buildBottomLevelAccelerationStructure(ground);
        buildBottomLevelAccelerationStructure(occluder);
        synchronizeAccelerationStructureBuild(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR);
        createTopLevelAccelerationStructure();
        synchronizeAccelerationStructureBuild(vk::PipelineStageFlagBits::eFragmentShader);
    }

    void createDescriptors() {
        const vk::DescriptorSetLayoutBinding binding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eAccelerationStructureKHR,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
        };
        descriptorSetLayout = device().createDescriptorSetLayout({
            .bindingCount = 1,
            .pBindings = &binding,
        });
        const vk::DescriptorPoolSize poolSize{
            .type = vk::DescriptorType::eAccelerationStructureKHR,
            .descriptorCount = 1,
        };
        descriptorPool = device().createDescriptorPool({
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &poolSize,
        });

        const vk::DescriptorSetLayout layout = *descriptorSetLayout;
        descriptorSets = device().allocateDescriptorSets({
            .descriptorPool = *descriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &layout,
        });
        const vk::AccelerationStructureKHR topLevelHandle = *topLevelAS.handle;
        const vk::WriteDescriptorSetAccelerationStructureKHR accelerationStructureWrite{
            .accelerationStructureCount = 1,
            .pAccelerationStructures = &topLevelHandle,
        };
        const vk::WriteDescriptorSet write{
            .pNext = &accelerationStructureWrite,
            .dstSet = *descriptorSets.front(),
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eAccelerationStructureKHR,
        };
        device().updateDescriptorSets({write}, {});
    }

    void createGraphicsPipeline() {
        shaderModules.clear();
        shaderModules.reserve(2);
        const std::filesystem::path shaderDirectory{SHADER_SPV_DIR};
        const auto vertexSpirv = readSpirvFile(shaderDirectory / "rayquery.vert.spv");
        const auto fragmentSpirv = readSpirvFile(shaderDirectory / "rayquery.frag.spv");
        shaderModules.emplace_back(device(), vk::ShaderModuleCreateInfo{
                                               .codeSize = vertexSpirv.size() * sizeof(std::uint32_t),
                                               .pCode = vertexSpirv.data(),
                                           });
        shaderModules.emplace_back(device(), vk::ShaderModuleCreateInfo{
                                               .codeSize = fragmentSpirv.size() * sizeof(std::uint32_t),
                                               .pCode = fragmentSpirv.data(),
                                           });

        const std::array shaderStages = {
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = *shaderModules[0],
                .pName = "main",
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = *shaderModules[1],
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
            .size = static_cast<std::uint32_t>(sizeof(ViewProjectionPushConstants)),
        };
        const vk::DescriptorSetLayout layout = *descriptorSetLayout;
        pipelineLayout = device().createPipelineLayout({
            .setLayoutCount = 1,
            .pSetLayouts = &layout,
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
                                                                     .renderPass = swapchainRenderPass(),
                                                                     .subpass = 0,
                                                                 });
    }

    void recordFrameCommands(vk::raii::CommandBuffer &commandBuffer, std::uint32_t imageIndex) override {
        const auto extent = swapchainExtent();
        const auto projection = glm::perspectiveRH_ZO(0.78539816339744830962f,
                                                       static_cast<float>(extent.width) / static_cast<float>(extent.height),
                                                       0.1f, 10.0f);
        const ViewProjectionPushConstants pushConstants{
            .viewProjection = projection * glm::lookAtRH(glm::vec3{2.7f, 2.4f, 3.7f},
                                                          glm::vec3{0.0f, 0.25f, 0.0f},
                                                          glm::vec3{0.0f, 1.0f, 0.0f}),
        };

        beginSwapchainRenderPass(commandBuffer, imageIndex);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
        const std::array descriptorSetHandles = {*descriptorSets.front()};
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0,
                                         descriptorSetHandles, {});
        commandBuffer.pushConstants(*pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0,
                                    static_cast<std::uint32_t>(sizeof(pushConstants)), &pushConstants);

        const std::array<const MeshResource *, 2> meshes = {&ground, &occluder};
        for (const auto *mesh : meshes) {
            const vk::Buffer vertexBuffer = *mesh->vertexBuffer.buffer;
            constexpr vk::DeviceSize vertexOffset = 0;
            commandBuffer.bindVertexBuffers(0, {vertexBuffer}, {vertexOffset});
            commandBuffer.bindIndexBuffer(*mesh->indexBuffer.buffer, 0, vk::IndexType::eUint32);
            commandBuffer.drawIndexed(mesh->indexCount, 1, 0, 0, 0);
        }
        commandBuffer.endRenderPass();
    }
};

int main() {
    try {
        RayQueryApplication application;
        application.run();
    } catch (const std::exception &error) {
        std::println("Error: {}", error.what());
        return 1;
    }
    return 0;
}
