#include <glm/ext/matrix_clip_space.hpp>
#include <glm/glm.hpp>
#include <imgui.h>
#include <tiny_obj_loader.h>

import vulkan;
import std;
import common;
import common.imgui_overlay;

#ifndef BUNNY_OBJ_PATH
#error "BUNNY_OBJ_PATH must be set by CMake (see examples/bvh/CMakeLists.txt)"
#endif

#ifndef SHADER_SPV_DIR
#error "SHADER_SPV_DIR must be set by CMake (see examples/bvh/CMakeLists.txt)"
#endif

namespace {

constexpr vk::Format outputFormat = vk::Format::eR8G8B8A8Unorm;
constexpr std::uint32_t workgroupSize = 8;
constexpr std::uint32_t leafTriangleLimit = 4;
constexpr std::uint32_t maxTraversalDepth = 60;
constexpr std::uint32_t maxVisibleBounds = 512;
constexpr std::uint32_t sahBucketCount = 12;
constexpr float cameraVerticalFovRadians = 0.78539816339744830962f;
constexpr float orbitRadiansPerPixel = 0.005f;
constexpr float minimumTriangleAreaSquared = 1.0e-20f;

enum class BvhBuildStrategy : std::uint32_t {
    median,
    midpoint,
    sah,
};

struct alignas(16) GpuNode {
    glm::vec4 boundsMinimum{};
    glm::vec4 boundsMaximum{};
    std::array<std::uint32_t, 4> data{};
};

struct alignas(16) GpuTriangle {
    glm::vec4 position0{};
    glm::vec4 position1{};
    glm::vec4 position2{};
    glm::vec4 normal0{};
    glm::vec4 normal1{};
    glm::vec4 normal2{};
};

struct BuildTriangle {
    GpuTriangle gpu;
    glm::vec3 boundsMinimum{};
    glm::vec3 boundsMaximum{};
    glm::vec3 centroid{};
};

struct CpuNode {
    GpuNode gpu;
    std::uint32_t depth = 0;
};

struct TriangleRangeBounds {
    glm::vec3 minimum{std::numeric_limits<float>::max()};
    glm::vec3 maximum{std::numeric_limits<float>::lowest()};
    std::uint32_t count = 0;
};

struct SahBucket {
    TriangleRangeBounds bounds;
};

struct BufferResource {
    vk::raii::DeviceMemory memory = nullptr;
    vk::raii::Buffer buffer = nullptr;
    vk::DeviceSize size = 0;
};

struct BvhCandidate {
    std::vector<CpuNode> nodes;
    std::vector<std::uint32_t> nodesPerDepth;
    BufferResource nodeBuffer;
    BufferResource triangleBuffer;
    std::uint32_t triangleCount = 0;
    std::uint32_t maxVisibleDepth = 0;
    double cpuBuildMilliseconds = 0.0;
};

struct ImageResource {
    vk::raii::DeviceMemory memory = nullptr;
    vk::raii::Image image = nullptr;
    vk::raii::ImageView imageView = nullptr;
};

struct alignas(16) RayPushConstants {
    glm::mat4 inverseViewProjection{1.0f};
    glm::uvec4 extent{0};
};

struct GpuStatistics {
    std::uint32_t nodeTests = 0;
    std::uint32_t triangleTests = 0;
};

struct StatisticsSlot {
    BufferResource deviceBuffer;
    BufferResource readbackBuffer;
    bool pending = false;
};

static_assert(sizeof(GpuNode) == 48);
static_assert(sizeof(GpuTriangle) == 96);
static_assert(sizeof(RayPushConstants) == 80);
static_assert(sizeof(GpuStatistics) == 8);

[[nodiscard]] std::uint32_t dispatchGroupCount(std::uint32_t invocationCount) {
    return (invocationCount + workgroupSize - 1u) / workgroupSize;
}

[[nodiscard]] glm::vec3 readPosition(const tinyobj::attrib_t &attributes, std::uint32_t index) {
    const auto offset = static_cast<std::size_t>(index) * 3;
    return {
        static_cast<float>(attributes.vertices[offset]),
        static_cast<float>(attributes.vertices[offset + 1]),
        static_cast<float>(attributes.vertices[offset + 2]),
    };
}

[[nodiscard]] std::vector<BuildTriangle> loadBunnyTriangles() {
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
    std::vector<std::array<std::uint32_t, 3>> sourceTriangles;
    for (const auto &shape : reader.GetShapes()) {
        std::size_t indexOffset = 0;
        for (const auto faceVertexCount : shape.mesh.num_face_vertices) {
            if (faceVertexCount != 3 || indexOffset + faceVertexCount > shape.mesh.indices.size()) {
                throw std::runtime_error("OBJ triangulation produced an invalid face");
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
            if (glm::dot(faceNormal, faceNormal) > minimumTriangleAreaSquared) {
                for (const auto vertex : triangle) {
                    normalSums[vertex] += faceNormal;
                }
                sourceTriangles.push_back(triangle);
            }
            indexOffset += faceVertexCount;
        }
        if (indexOffset != shape.mesh.indices.size()) {
            throw std::runtime_error("OBJ shape index count does not match its faces");
        }
    }
    if (sourceTriangles.empty()) {
        throw std::runtime_error("OBJ contains no non-degenerate triangles");
    }
    if (sourceTriangles.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("OBJ has too many triangles for uint32 indices");
    }

    const auto extent = maximum - minimum;
    const auto largestExtent = std::max({extent.x, extent.y, extent.z});
    if (largestExtent <= std::numeric_limits<float>::epsilon()) {
        throw std::runtime_error("OBJ has a degenerate bounding box");
    }
    const auto center = (minimum + maximum) * 0.5f;
    const auto scale = 1.8f / largestExtent;
    for (auto &position : positions) {
        position = (position - center) * scale;
    }

    std::vector<glm::vec3> normals(vertexCount, glm::vec3{0.0f, 1.0f, 0.0f});
    for (std::size_t index = 0; index < vertexCount; ++index) {
        const auto squaredLength = glm::dot(normalSums[index], normalSums[index]);
        if (squaredLength > minimumTriangleAreaSquared) {
            normals[index] = glm::normalize(normalSums[index]);
        }
    }

    std::vector<BuildTriangle> triangles;
    triangles.reserve(sourceTriangles.size());
    for (const auto &source : sourceTriangles) {
        const auto p0 = positions[source[0]];
        const auto p1 = positions[source[1]];
        const auto p2 = positions[source[2]];
        BuildTriangle triangle;
        triangle.gpu.position0 = glm::vec4{p0, 1.0f};
        triangle.gpu.position1 = glm::vec4{p1, 1.0f};
        triangle.gpu.position2 = glm::vec4{p2, 1.0f};
        triangle.gpu.normal0 = glm::vec4{normals[source[0]], 0.0f};
        triangle.gpu.normal1 = glm::vec4{normals[source[1]], 0.0f};
        triangle.gpu.normal2 = glm::vec4{normals[source[2]], 0.0f};
        triangle.boundsMinimum = glm::min(p0, glm::min(p1, p2));
        triangle.boundsMaximum = glm::max(p0, glm::max(p1, p2));
        triangle.centroid = (p0 + p1 + p2) / 3.0f;
        triangles.push_back(triangle);
    }

    std::println("Loaded bunny.obj: {} triangles", triangles.size());
    return triangles;
}

[[nodiscard]] std::string_view strategyName(BvhBuildStrategy strategy) {
    switch (strategy) {
    case BvhBuildStrategy::median:
        return "Longest-axis median";
    case BvhBuildStrategy::midpoint:
        return "Centroid midpoint";
    case BvhBuildStrategy::sah:
        return "12-bucket SAH";
    }
    return "Unknown";
}

[[nodiscard]] std::uint32_t longestCentroidAxis(const glm::vec3 &centroidMinimum,
                                                const glm::vec3 &centroidMaximum) {
    const auto centroidExtent = centroidMaximum - centroidMinimum;
    std::uint32_t axis = 0;
    if (centroidExtent.y > centroidExtent.x) {
        axis = 1;
    }
    if (centroidExtent.z > centroidExtent[axis]) {
        axis = 2;
    }
    return axis;
}

[[nodiscard]] float surfaceArea(const TriangleRangeBounds &bounds) {
    const auto extent = glm::max(bounds.maximum - bounds.minimum, glm::vec3{0.0f});
    return 2.0f * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x);
}

[[nodiscard]] std::uint32_t centroidBucket(float centroid,
                                          float centroidMinimum,
                                          float centroidExtent) {
    const auto normalized = std::clamp((centroid - centroidMinimum) / centroidExtent, 0.0f, 1.0f);
    const auto bucket = static_cast<std::uint32_t>(normalized * static_cast<float>(sahBucketCount));
    return std::min(bucket, sahBucketCount - 1);
}

[[nodiscard]] std::uint32_t medianSplit(std::vector<BuildTriangle> &triangles,
                                       std::uint32_t begin,
                                       std::uint32_t end,
                                       std::uint32_t axis) {
    const auto middle = begin + (end - begin) / 2;
    std::nth_element(triangles.begin() + begin, triangles.begin() + middle, triangles.begin() + end,
                     [axis](const BuildTriangle &left, const BuildTriangle &right) {
                         return left.centroid[axis] < right.centroid[axis];
                     });
    return middle;
}

[[nodiscard]] std::uint32_t chooseSahSplit(std::vector<BuildTriangle> &triangles,
                                          std::uint32_t begin,
                                          std::uint32_t end,
                                          const glm::vec3 &centroidMinimum,
                                          const glm::vec3 &centroidMaximum,
                                          bool &splitFound) {
    constexpr float noSplitCost = std::numeric_limits<float>::max();
    auto bestCost = noSplitCost;
    auto bestAxis = 0u;
    auto bestBucket = 0u;
    splitFound = false;

    for (std::uint32_t axis = 0; axis < 3; ++axis) {
        const auto centroidExtent = centroidMaximum[axis] - centroidMinimum[axis];
        if (centroidExtent <= std::numeric_limits<float>::epsilon()) {
            continue;
        }

        std::array<SahBucket, sahBucketCount> buckets{};
        for (auto index = begin; index < end; ++index) {
            auto &bucket = buckets[centroidBucket(triangles[index].centroid[axis],
                                                  centroidMinimum[axis], centroidExtent)];
            bucket.bounds.minimum = glm::min(bucket.bounds.minimum, triangles[index].boundsMinimum);
            bucket.bounds.maximum = glm::max(bucket.bounds.maximum, triangles[index].boundsMaximum);
            ++bucket.bounds.count;
        }

        std::array<TriangleRangeBounds, sahBucketCount> leftBounds{};
        std::array<TriangleRangeBounds, sahBucketCount> rightBounds{};
        auto prefix = TriangleRangeBounds{};
        for (std::uint32_t bucket = 0; bucket < sahBucketCount; ++bucket) {
            if (buckets[bucket].bounds.count > 0) {
                prefix.minimum = glm::min(prefix.minimum, buckets[bucket].bounds.minimum);
                prefix.maximum = glm::max(prefix.maximum, buckets[bucket].bounds.maximum);
                prefix.count += buckets[bucket].bounds.count;
            }
            leftBounds[bucket] = prefix;
        }
        auto suffix = TriangleRangeBounds{};
        for (auto bucket = sahBucketCount; bucket-- > 0;) {
            if (buckets[bucket].bounds.count > 0) {
                suffix.minimum = glm::min(suffix.minimum, buckets[bucket].bounds.minimum);
                suffix.maximum = glm::max(suffix.maximum, buckets[bucket].bounds.maximum);
                suffix.count += buckets[bucket].bounds.count;
            }
            rightBounds[bucket] = suffix;
        }

        for (std::uint32_t bucket = 0; bucket + 1 < sahBucketCount; ++bucket) {
            const auto &left = leftBounds[bucket];
            const auto &right = rightBounds[bucket + 1];
            if (left.count == 0 || right.count == 0) {
                continue;
            }
            const auto cost = surfaceArea(left) * static_cast<float>(left.count) +
                              surfaceArea(right) * static_cast<float>(right.count);
            if (cost < bestCost) {
                bestCost = cost;
                bestAxis = axis;
                bestBucket = bucket;
                splitFound = true;
            }
        }
    }

    if (!splitFound) {
        return begin;
    }

    const auto axisMinimum = centroidMinimum[bestAxis];
    const auto axisExtent = centroidMaximum[bestAxis] - axisMinimum;
    const auto partition = std::partition(triangles.begin() + begin, triangles.begin() + end,
                                          [bestAxis, bestBucket, axisMinimum,
                                           axisExtent](const BuildTriangle &triangle) {
                                              return centroidBucket(triangle.centroid[bestAxis], axisMinimum,
                                                                    axisExtent) <= bestBucket;
                                          });
    return static_cast<std::uint32_t>(partition - triangles.begin());
}

[[nodiscard]] std::uint32_t chooseSplit(std::vector<BuildTriangle> &triangles,
                                        std::uint32_t begin,
                                        std::uint32_t end,
                                        const glm::vec3 &centroidMinimum,
                                        const glm::vec3 &centroidMaximum,
                                        std::uint32_t depth,
                                        BvhBuildStrategy strategy) {
    const auto medianAxis = longestCentroidAxis(centroidMinimum, centroidMaximum);
    auto middle = begin;
    switch (strategy) {
    case BvhBuildStrategy::median:
        return medianSplit(triangles, begin, end, medianAxis);
    case BvhBuildStrategy::midpoint: {
        const auto midpoint = (centroidMinimum[medianAxis] + centroidMaximum[medianAxis]) * 0.5f;
        const auto partition = std::partition(triangles.begin() + begin, triangles.begin() + end,
                                              [medianAxis, midpoint](const BuildTriangle &triangle) {
                                                  return triangle.centroid[medianAxis] < midpoint;
                                              });
        middle = static_cast<std::uint32_t>(partition - triangles.begin());
        break;
    }
    case BvhBuildStrategy::sah: {
        bool splitFound = false;
        middle = chooseSahSplit(triangles, begin, end, centroidMinimum, centroidMaximum, splitFound);
        if (!splitFound) {
            middle = begin;
        }
        break;
    }
    }

    const auto remainingLevels = maxTraversalDepth - depth - 1;
    const auto maximumChildTriangleCount = std::min<std::uint64_t>(
        std::numeric_limits<std::uint32_t>::max(),
        static_cast<std::uint64_t>(leafTriangleLimit) * (std::uint64_t{1} << remainingLevels));
    if (middle == begin || middle == end || middle - begin > maximumChildTriangleCount ||
        end - middle > maximumChildTriangleCount) {
        middle = medianSplit(triangles, begin, end, medianAxis);
        if (middle - begin > maximumChildTriangleCount || end - middle > maximumChildTriangleCount) {
            throw std::runtime_error("BVH cannot satisfy the shader traversal stack depth");
        }
    }
    return middle;
}

std::uint32_t buildNode(std::vector<BuildTriangle> &triangles,
                        std::vector<CpuNode> &nodes,
                        std::vector<std::uint32_t> &nodesPerDepth,
                        std::uint32_t begin,
                        std::uint32_t end,
                        std::uint32_t depth,
                        BvhBuildStrategy strategy) {
    if (depth >= maxTraversalDepth && end - begin > leafTriangleLimit) {
        throw std::runtime_error("BVH exceeded the shader traversal stack depth");
    }

    glm::vec3 boundsMinimum{std::numeric_limits<float>::max()};
    glm::vec3 boundsMaximum{std::numeric_limits<float>::lowest()};
    glm::vec3 centroidMinimum{std::numeric_limits<float>::max()};
    glm::vec3 centroidMaximum{std::numeric_limits<float>::lowest()};
    for (auto index = begin; index < end; ++index) {
        boundsMinimum = glm::min(boundsMinimum, triangles[index].boundsMinimum);
        boundsMaximum = glm::max(boundsMaximum, triangles[index].boundsMaximum);
        centroidMinimum = glm::min(centroidMinimum, triangles[index].centroid);
        centroidMaximum = glm::max(centroidMaximum, triangles[index].centroid);
    }

    const auto nodeIndex = static_cast<std::uint32_t>(nodes.size());
    CpuNode node;
    node.gpu.boundsMinimum = glm::vec4{boundsMinimum, 0.0f};
    node.gpu.boundsMaximum = glm::vec4{boundsMaximum, 0.0f};
    node.depth = depth;
    nodes.push_back(node);
    if (nodesPerDepth.size() <= depth) {
        nodesPerDepth.resize(depth + 1, 0);
    }
    ++nodesPerDepth[depth];

    const auto triangleCount = end - begin;
    if (triangleCount <= leafTriangleLimit) {
        nodes[nodeIndex].gpu.data = {0, 0, begin, triangleCount};
        return nodeIndex;
    }

    const auto middle = chooseSplit(triangles, begin, end, centroidMinimum, centroidMaximum, depth, strategy);

    const auto leftChild = buildNode(triangles, nodes, nodesPerDepth, begin, middle, depth + 1, strategy);
    const auto rightChild = buildNode(triangles, nodes, nodesPerDepth, middle, end, depth + 1, strategy);
    nodes[nodeIndex].gpu.data = {leftChild, rightChild, 0, 0};
    return nodeIndex;
}

}  // namespace

class BvhApplication final : public WindowedVulkanApp {
public:
    BvhApplication()
        : WindowedVulkanApp({
              .window = {
                  .width = 1120,
                  .height = 760,
                  .title = "Vulkan Software BVH",
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
    ImageResource outputImage;
    vk::raii::Sampler outputSampler = nullptr;
    BufferResource nodeBuffer;
    BufferResource triangleBuffer;

    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> descriptorSets;

    vk::raii::ShaderModule computeShader = nullptr;
    vk::raii::ShaderModule fullscreenVertexShader = nullptr;
    vk::raii::ShaderModule fullscreenFragmentShader = nullptr;
    vk::raii::PipelineLayout computePipelineLayout = nullptr;
    vk::raii::PipelineLayout presentationPipelineLayout = nullptr;
    vk::raii::Pipeline computePipeline = nullptr;
    vk::raii::Pipeline presentationPipeline = nullptr;

    std::vector<CpuNode> cpuNodes;
    std::vector<std::uint32_t> nodesPerDepth;
    std::vector<BuildTriangle> sourceTriangles;
    std::vector<StatisticsSlot> statisticsSlots;
    std::uint32_t triangleCount = 0;
    std::uint32_t maxVisibleDepth = 0;
    int selectedDepth = 3;
    BvhBuildStrategy selectedStrategy = BvhBuildStrategy::median;
    BvhBuildStrategy activeStrategy = BvhBuildStrategy::median;
    double cpuBuildMilliseconds = 0.0;
    std::string buildError;
    GpuStatistics statistics{};
    bool statisticsValid = false;
    bool outputImageInitialized = false;

    OrbitCamera camera;
    std::optional<GlfwCursorPosition> previousDragCursor;
    bool resetKeyWasPressed = false;
    bool showBounds = true;
    ImGuiVulkanOverlay overlay;

    bool supportsAdditionalDeviceRequirements(const vk::raii::PhysicalDevice &candidate) const override {
        const auto formatProperties = candidate.getFormatProperties(outputFormat);
        const auto requiredFormatFeatures =
            vk::FormatFeatureFlagBits::eStorageImage | vk::FormatFeatureFlagBits::eSampledImage;
        return (formatProperties.optimalTilingFeatures & requiredFormatFeatures) == requiredFormatFeatures;
    }

    void createApplicationResources() override {
        createBvhBuffers();
        createOutputImage();
        createOutputSampler();
        createStatisticsBuffers();
        createDescriptors();
        createPipelineLayouts();
        createShaderModules();
        createComputePipeline();
        createPresentationPipeline();
        overlay.initialize(makeOverlayConfig());
    }

    void releaseSwapchainDependentResources() override {
        overlay.releaseRenderer();
        presentationPipeline = nullptr;
        outputImage.imageView = nullptr;
        outputImage.image = nullptr;
        outputImage.memory = nullptr;
        outputImageInitialized = false;
    }

    void recreateSwapchainDependentResources() override {
        createOutputImage();
        updateImageDescriptors();
        createPresentationPipeline();
        overlay.recreateRenderer(swapchainRenderPass(), imguiMinImageCount(),
                                 static_cast<std::uint32_t>(swapchainImages().size()));
    }

    void recordFrameCommands(vk::raii::CommandBuffer &commandBuffer, std::uint32_t imageIndex) override {
        const auto frameIndex = currentFrameIndex();
        readCompletedStatistics(frameIndex);

        overlay.beginFrame();
        drawControlPanel();
        updateCameraFromInput();

        recordCompute(commandBuffer, frameIndex);
        recordPresentation(commandBuffer, imageIndex, frameIndex);
    }

    [[nodiscard]] BufferResource createBuffer(vk::DeviceSize size,
                                              vk::BufferUsageFlags usage,
                                              vk::MemoryPropertyFlags memoryProperties) {
        if (size == 0) {
            throw std::invalid_argument("Cannot create a zero-sized buffer");
        }

        BufferResource resource;
        resource.size = size;
        resource.buffer = device().createBuffer({
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
        });
        const auto requirements = resource.buffer.getMemoryRequirements();
        resource.memory = vk::raii::DeviceMemory(device(), {
            .allocationSize = requirements.size,
            .memoryTypeIndex = findMemoryTypeIndex(requirements, memoryProperties),
        });
        resource.buffer.bindMemory(*resource.memory, 0);
        return resource;
    }

    void uploadHostBuffer(const BufferResource &resource, const void *data, vk::DeviceSize size) {
        void *mapped = resource.memory.mapMemory(0, size);
        std::memcpy(mapped, data, static_cast<std::size_t>(size));
        resource.memory.unmapMemory();
    }

    void uploadDeviceBuffer(BufferResource &destination, const void *data, vk::DeviceSize size) {
        auto staging = createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc,
                                    vk::MemoryPropertyFlagBits::eHostVisible |
                                        vk::MemoryPropertyFlagBits::eHostCoherent);
        uploadHostBuffer(staging, data, size);

        const vk::CommandBufferAllocateInfo allocateInfo{
            .commandPool = *commandPool(),
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        };
        auto commandBuffers = device().allocateCommandBuffers(allocateInfo);
        auto &commandBuffer = commandBuffers.front();
        commandBuffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        commandBuffer.copyBuffer(*staging.buffer, *destination.buffer, {vk::BufferCopy{
                                         .srcOffset = 0,
                                         .dstOffset = 0,
                                         .size = size,
                                     }});
        const vk::BufferMemoryBarrier uploadBarrier{
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .buffer = *destination.buffer,
            .offset = 0,
            .size = size,
        };
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                      vk::PipelineStageFlagBits::eComputeShader, {}, {}, {uploadBarrier}, {});
        commandBuffer.end();

        const vk::CommandBuffer commandBufferHandle = *commandBuffer;
        const vk::SubmitInfo submitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBufferHandle,
        };
        graphicsQueue().submit(submitInfo);
        graphicsQueue().waitIdle();
    }

    void createBvhBuffers() {
        auto candidate = buildBvhCandidate(BvhBuildStrategy::median);
        installBvhCandidate(std::move(candidate), BvhBuildStrategy::median);
    }

    [[nodiscard]] BvhCandidate buildBvhCandidate(BvhBuildStrategy strategy) {
        if (sourceTriangles.empty()) {
            sourceTriangles = loadBunnyTriangles();
        }
        if (sourceTriangles.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("BVH triangle count exceeds uint32 range");
        }
        const auto buildStart = std::chrono::steady_clock::now();
        auto triangles = sourceTriangles;
        BvhCandidate candidate;
        candidate.triangleCount = static_cast<std::uint32_t>(triangles.size());
        candidate.nodes.reserve(triangles.size() * 2);
        (void)buildNode(triangles, candidate.nodes, candidate.nodesPerDepth, 0,
                        candidate.triangleCount, 0, strategy);

        while (candidate.maxVisibleDepth + 1 < candidate.nodesPerDepth.size() &&
               candidate.nodesPerDepth[candidate.maxVisibleDepth + 1] <= maxVisibleBounds) {
            ++candidate.maxVisibleDepth;
        }
        candidate.cpuBuildMilliseconds =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - buildStart).count();

        std::vector<GpuNode> gpuNodes;
        gpuNodes.reserve(candidate.nodes.size());
        for (const auto &node : candidate.nodes) {
            gpuNodes.push_back(node.gpu);
        }
        std::vector<GpuTriangle> gpuTriangles;
        gpuTriangles.reserve(triangles.size());
        for (const auto &triangle : triangles) {
            gpuTriangles.push_back(triangle.gpu);
        }

        const auto maxStorageRange = physicalDevice().getProperties().limits.maxStorageBufferRange;
        const auto nodeBytes = static_cast<vk::DeviceSize>(gpuNodes.size() * sizeof(GpuNode));
        const auto triangleBytes = static_cast<vk::DeviceSize>(gpuTriangles.size() * sizeof(GpuTriangle));
        if (nodeBytes > maxStorageRange || triangleBytes > maxStorageRange) {
            throw std::runtime_error("Bunny BVH data exceeds maxStorageBufferRange");
        }

        candidate.nodeBuffer = createBuffer(
            nodeBytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        candidate.triangleBuffer = createBuffer(
            triangleBytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        uploadDeviceBuffer(candidate.nodeBuffer, gpuNodes.data(), nodeBytes);
        uploadDeviceBuffer(candidate.triangleBuffer, gpuTriangles.data(), triangleBytes);
        return candidate;
    }

    void installBvhCandidate(BvhCandidate candidate, BvhBuildStrategy strategy) {
        if (!descriptorSets.empty()) {
            updateBvhBufferDescriptors(candidate.nodeBuffer, candidate.triangleBuffer);
        }

        nodeBuffer = std::move(candidate.nodeBuffer);
        triangleBuffer = std::move(candidate.triangleBuffer);
        cpuNodes = std::move(candidate.nodes);
        nodesPerDepth = std::move(candidate.nodesPerDepth);
        triangleCount = candidate.triangleCount;
        maxVisibleDepth = candidate.maxVisibleDepth;
        selectedDepth = static_cast<int>(std::min(3u, maxVisibleDepth));
        activeStrategy = strategy;
        selectedStrategy = strategy;
        cpuBuildMilliseconds = candidate.cpuBuildMilliseconds;
        statistics = {};
        statisticsValid = false;
        for (auto &slot : statisticsSlots) {
            slot.pending = false;
        }
        buildError.clear();
        std::println("Built {} BVH: {} nodes, {} triangles, depth {}, CPU {:.3f} ms",
                     strategyName(activeStrategy), cpuNodes.size(), triangleCount, nodesPerDepth.size() - 1,
                     cpuBuildMilliseconds);
    }

    void rebuildBvh() {
        try {
            // Descriptors and recorded submissions can still reference the current buffers.
            device().waitIdle();
            auto candidate = buildBvhCandidate(selectedStrategy);
            installBvhCandidate(std::move(candidate), selectedStrategy);
        } catch (const std::exception &error) {
            buildError = error.what();
        }
    }

    void createOutputImage() {
        const auto extent = swapchainExtent();
        outputImage.image = device().createImage({
            .imageType = vk::ImageType::e2D,
            .format = outputFormat,
            .extent = {extent.width, extent.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined,
        });

        const auto requirements = outputImage.image.getMemoryRequirements();
        outputImage.memory = vk::raii::DeviceMemory(device(), {
            .allocationSize = requirements.size,
            .memoryTypeIndex = findMemoryTypeIndex(requirements, vk::MemoryPropertyFlagBits::eDeviceLocal),
        });
        outputImage.image.bindMemory(*outputImage.memory, 0);
        outputImage.imageView = device().createImageView({
            .image = *outputImage.image,
            .viewType = vk::ImageViewType::e2D,
            .format = outputFormat,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        });
        outputImageInitialized = false;
    }

    void createOutputSampler() {
        outputSampler = device().createSampler({
            .magFilter = vk::Filter::eNearest,
            .minFilter = vk::Filter::eNearest,
            .mipmapMode = vk::SamplerMipmapMode::eNearest,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
            .maxAnisotropy = 1.0f,
            .compareOp = vk::CompareOp::eAlways,
            .minLod = 0.0f,
            .maxLod = 0.0f,
            .borderColor = vk::BorderColor::eFloatOpaqueBlack,
            .unnormalizedCoordinates = false,
        });
    }

    void createStatisticsBuffers() {
        statisticsSlots.reserve(framesInFlight());
        for (std::uint32_t index = 0; index < framesInFlight(); ++index) {
            StatisticsSlot slot;
            slot.deviceBuffer = createBuffer(sizeof(GpuStatistics),
                                             vk::BufferUsageFlagBits::eStorageBuffer |
                                                 vk::BufferUsageFlagBits::eTransferDst |
                                                 vk::BufferUsageFlagBits::eTransferSrc,
                                             vk::MemoryPropertyFlagBits::eDeviceLocal);
            slot.readbackBuffer = createBuffer(sizeof(GpuStatistics), vk::BufferUsageFlagBits::eTransferDst,
                                               vk::MemoryPropertyFlagBits::eHostVisible |
                                                   vk::MemoryPropertyFlagBits::eHostCoherent);
            statisticsSlots.push_back(std::move(slot));
        }
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
            vk::DescriptorSetLayoutBinding{
                .binding = 2,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eCompute,
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 3,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eCompute,
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 4,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eCompute,
            },
        };
        descriptorSetLayout = device().createDescriptorSetLayout({
            .bindingCount = static_cast<std::uint32_t>(bindings.size()),
            .pBindings = bindings.data(),
        });

        const auto frameCount = framesInFlight();
        const std::array poolSizes = {
            vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageImage, .descriptorCount = frameCount},
            vk::DescriptorPoolSize{.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = frameCount},
            vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 3 * frameCount},
        };
        descriptorPool = device().createDescriptorPool({
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = frameCount,
            .poolSizeCount = static_cast<std::uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data(),
        });
        const std::vector<vk::DescriptorSetLayout> layouts(frameCount, *descriptorSetLayout);
        descriptorSets = device().allocateDescriptorSets({
            .descriptorPool = *descriptorPool,
            .descriptorSetCount = frameCount,
            .pSetLayouts = layouts.data(),
        });

        for (std::uint32_t index = 0; index < frameCount; ++index) {
            const vk::DescriptorImageInfo storageImageInfo{
                .imageView = *outputImage.imageView,
                .imageLayout = vk::ImageLayout::eGeneral,
            };
            const vk::DescriptorImageInfo sampledImageInfo{
                .sampler = *outputSampler,
                .imageView = *outputImage.imageView,
                .imageLayout = vk::ImageLayout::eGeneral,
            };
            const vk::DescriptorBufferInfo nodeInfo{
                .buffer = *nodeBuffer.buffer,
                .offset = 0,
                .range = nodeBuffer.size,
            };
            const vk::DescriptorBufferInfo triangleInfo{
                .buffer = *triangleBuffer.buffer,
                .offset = 0,
                .range = triangleBuffer.size,
            };
            const vk::DescriptorBufferInfo statisticsInfo{
                .buffer = *statisticsSlots[index].deviceBuffer.buffer,
                .offset = 0,
                .range = sizeof(GpuStatistics),
            };
            const std::array writes = {
                vk::WriteDescriptorSet{
                    .dstSet = *descriptorSets[index],
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageImage,
                    .pImageInfo = &storageImageInfo,
                },
                vk::WriteDescriptorSet{
                    .dstSet = *descriptorSets[index],
                    .dstBinding = 1,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo = &sampledImageInfo,
                },
                vk::WriteDescriptorSet{
                    .dstSet = *descriptorSets[index],
                    .dstBinding = 2,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo = &nodeInfo,
                },
                vk::WriteDescriptorSet{
                    .dstSet = *descriptorSets[index],
                    .dstBinding = 3,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo = &triangleInfo,
                },
                vk::WriteDescriptorSet{
                    .dstSet = *descriptorSets[index],
                    .dstBinding = 4,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo = &statisticsInfo,
                },
            };
            device().updateDescriptorSets(writes, {});
        }
    }

    void updateBvhBufferDescriptors(const BufferResource &newNodeBuffer,
                                    const BufferResource &newTriangleBuffer) {
        for (const auto &descriptorSet : descriptorSets) {
            const vk::DescriptorBufferInfo nodeInfo{
                .buffer = *newNodeBuffer.buffer,
                .offset = 0,
                .range = newNodeBuffer.size,
            };
            const vk::DescriptorBufferInfo triangleInfo{
                .buffer = *newTriangleBuffer.buffer,
                .offset = 0,
                .range = newTriangleBuffer.size,
            };
            const std::array writes = {
                vk::WriteDescriptorSet{
                    .dstSet = *descriptorSet,
                    .dstBinding = 2,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo = &nodeInfo,
                },
                vk::WriteDescriptorSet{
                    .dstSet = *descriptorSet,
                    .dstBinding = 3,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo = &triangleInfo,
                },
            };
            device().updateDescriptorSets(writes, {});
        }
    }

    void updateImageDescriptors() {
        for (const auto &descriptorSet : descriptorSets) {
            const vk::DescriptorImageInfo storageImageInfo{
                .imageView = *outputImage.imageView,
                .imageLayout = vk::ImageLayout::eGeneral,
            };
            const vk::DescriptorImageInfo sampledImageInfo{
                .sampler = *outputSampler,
                .imageView = *outputImage.imageView,
                .imageLayout = vk::ImageLayout::eGeneral,
            };
            const std::array writes = {
                vk::WriteDescriptorSet{
                    .dstSet = *descriptorSet,
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageImage,
                    .pImageInfo = &storageImageInfo,
                },
                vk::WriteDescriptorSet{
                    .dstSet = *descriptorSet,
                    .dstBinding = 1,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo = &sampledImageInfo,
                },
            };
            device().updateDescriptorSets(writes, {});
        }
    }

    void createPipelineLayouts() {
        const vk::DescriptorSetLayout rawLayout = *descriptorSetLayout;
        const vk::PushConstantRange computePushRange{
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset = 0,
            .size = sizeof(RayPushConstants),
        };
        computePipelineLayout = device().createPipelineLayout({
            .setLayoutCount = 1,
            .pSetLayouts = &rawLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &computePushRange,
        });
        presentationPipelineLayout = device().createPipelineLayout({
            .setLayoutCount = 1,
            .pSetLayouts = &rawLayout,
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
        computeShader = loadShaderModule(shaderDirectory / "bvh.comp.spv");
        fullscreenVertexShader = loadShaderModule(shaderDirectory / "fullscreen.vert.spv");
        fullscreenFragmentShader = loadShaderModule(shaderDirectory / "fullscreen.frag.spv");
    }

    void createComputePipeline() {
        computePipeline = device().createComputePipeline(nullptr, {
            .stage = {
                .stage = vk::ShaderStageFlagBits::eCompute,
                .module = *computeShader,
                .pName = "main",
            },
            .layout = *computePipelineLayout,
        });
    }

    void createPresentationPipeline() {
        const std::array stages = {
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = *fullscreenVertexShader,
                .pName = "main",
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = *fullscreenFragmentShader,
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
            .cullMode = vk::CullModeFlagBits::eNone,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .lineWidth = 1.0f,
        };
        const vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
        };
        const vk::PipelineDepthStencilStateCreateInfo depthStencil{};
        const vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
        };
        const vk::PipelineColorBlendStateCreateInfo colorBlending{
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment,
        };
        presentationPipeline = device().createGraphicsPipeline(nullptr, {
            .stageCount = static_cast<std::uint32_t>(stages.size()),
            .pStages = stages.data(),
            .pVertexInputState = &vertexInput,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil,
            .pColorBlendState = &colorBlending,
            .layout = *presentationPipelineLayout,
            .renderPass = swapchainRenderPass(),
            .subpass = 0,
        });
    }

    void readCompletedStatistics(std::uint32_t frameIndex) {
        auto &slot = statisticsSlots.at(frameIndex);
        if (!slot.pending) {
            return;
        }

        const auto *mapped = static_cast<const GpuStatistics *>(slot.readbackBuffer.memory.mapMemory(
            0, sizeof(GpuStatistics)));
        statistics = *mapped;
        slot.readbackBuffer.memory.unmapMemory();
        statisticsValid = true;
        slot.pending = false;
    }

    void recordCompute(vk::raii::CommandBuffer &commandBuffer, std::uint32_t frameIndex) {
        auto &slot = statisticsSlots.at(frameIndex);
        commandBuffer.fillBuffer(*slot.deviceBuffer.buffer, 0, sizeof(GpuStatistics), 0);
        const vk::BufferMemoryBarrier statisticsResetBarrier{
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .buffer = *slot.deviceBuffer.buffer,
            .offset = 0,
            .size = sizeof(GpuStatistics),
        };
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                      vk::PipelineStageFlagBits::eComputeShader, {}, {},
                                      {statisticsResetBarrier}, {});

        const vk::ImageMemoryBarrier outputForWrite{
            .srcAccessMask = outputImageInitialized ? vk::AccessFlags{vk::AccessFlagBits::eShaderRead}
                                                    : vk::AccessFlags{},
            .dstAccessMask = vk::AccessFlagBits::eShaderWrite,
            .oldLayout = outputImageInitialized ? vk::ImageLayout::eGeneral : vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = *outputImage.image,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        const auto outputSourceStage = outputImageInitialized
                                           ? vk::PipelineStageFlags{vk::PipelineStageFlagBits::eFragmentShader}
                                           : vk::PipelineStageFlags{vk::PipelineStageFlagBits::eTopOfPipe};
        commandBuffer.pipelineBarrier(outputSourceStage, vk::PipelineStageFlagBits::eComputeShader,
                                      {}, {}, {}, {outputForWrite});
        outputImageInitialized = true;

        const auto extent = swapchainExtent();
        const auto projection = glm::perspectiveRH_ZO(
            cameraVerticalFovRadians,
            static_cast<float>(extent.width) / static_cast<float>(std::max(extent.height, 1u)),
            0.1f, 20.0f);
        const RayPushConstants pushConstants{
            .inverseViewProjection = glm::inverse(projection * camera.viewMatrix()),
            .extent = {extent.width, extent.height, 0, 0},
        };
        const vk::DescriptorSet descriptorSet = *descriptorSets.at(frameIndex);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *computePipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computePipelineLayout, 0,
                                         {descriptorSet}, {});
        commandBuffer.pushConstants(*computePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                                    sizeof(pushConstants), &pushConstants);
        commandBuffer.dispatch(dispatchGroupCount(extent.width), dispatchGroupCount(extent.height), 1);

        const vk::ImageMemoryBarrier outputForRead{
            .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = *outputImage.image,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                      vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, {outputForRead});

        const vk::BufferMemoryBarrier statisticsForCopy{
            .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
            .dstAccessMask = vk::AccessFlagBits::eTransferRead,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .buffer = *slot.deviceBuffer.buffer,
            .offset = 0,
            .size = sizeof(GpuStatistics),
        };
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                      vk::PipelineStageFlagBits::eTransfer, {}, {},
                                      {statisticsForCopy}, {});
        commandBuffer.copyBuffer(*slot.deviceBuffer.buffer, *slot.readbackBuffer.buffer, {vk::BufferCopy{
                                         .srcOffset = 0,
                                         .dstOffset = 0,
                                         .size = sizeof(GpuStatistics),
                                     }});
        const vk::BufferMemoryBarrier readbackForHost{
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eHostRead,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .buffer = *slot.readbackBuffer.buffer,
            .offset = 0,
            .size = sizeof(GpuStatistics),
        };
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                      vk::PipelineStageFlagBits::eHost, {}, {}, {readbackForHost}, {});
        slot.pending = true;
    }

    void recordPresentation(vk::raii::CommandBuffer &commandBuffer,
                           std::uint32_t imageIndex,
                           std::uint32_t frameIndex) {
        const auto extent = swapchainExtent();
        const auto viewProjection = glm::perspectiveRH_ZO(
                                        cameraVerticalFovRadians,
                                        static_cast<float>(extent.width) /
                                            static_cast<float>(std::max(extent.height, 1u)),
                                        0.1f, 20.0f) *
                                    camera.viewMatrix();

        beginSwapchainRenderPass(commandBuffer, imageIndex);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *presentationPipeline);
        const vk::DescriptorSet descriptorSet = *descriptorSets.at(frameIndex);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *presentationPipelineLayout, 0,
                                         {descriptorSet}, {});
        commandBuffer.draw(3, 1, 0, 0);
        drawBoundsOverlay(viewProjection);
        overlay.render(commandBuffer);
        commandBuffer.endRenderPass();
    }

    void drawBoundsOverlay(const glm::mat4 &viewProjection) const {
        if (!showBounds || cpuNodes.empty()) {
            return;
        }

        auto *drawList = ImGui::GetBackgroundDrawList();
        const auto displaySize = ImGui::GetIO().DisplaySize;
        const auto color = ImColor::HSV(std::fmod(static_cast<float>(selectedDepth) * 0.11f, 1.0f),
                                        0.75f, 1.0f, 0.9f);
        constexpr std::array<std::array<std::uint32_t, 2>, 12> edges = {{
            {{0, 1}}, {{0, 2}}, {{0, 4}}, {{1, 3}}, {{1, 5}}, {{2, 3}},
            {{2, 6}}, {{3, 7}}, {{4, 5}}, {{4, 6}}, {{5, 7}}, {{6, 7}},
        }};

        const auto project = [&](glm::vec3 position) -> std::optional<ImVec2> {
            const auto clip = viewProjection * glm::vec4{position, 1.0f};
            if (clip.w <= 1.0e-5f || clip.z < 0.0f || clip.z > clip.w) {
                return std::nullopt;
            }
            const auto ndc = glm::vec2{clip.x, clip.y} / clip.w;
            return ImVec2{(ndc.x * 0.5f + 0.5f) * displaySize.x,
                          (0.5f - ndc.y * 0.5f) * displaySize.y};
        };

        for (const auto &node : cpuNodes) {
            if (node.depth != static_cast<std::uint32_t>(selectedDepth)) {
                continue;
            }
            const auto minimum = glm::vec3{node.gpu.boundsMinimum};
            const auto maximum = glm::vec3{node.gpu.boundsMaximum};
            std::array<glm::vec3, 8> corners{};
            for (std::uint32_t corner = 0; corner < corners.size(); ++corner) {
                corners[corner] = {
                    (corner & 1u) != 0 ? maximum.x : minimum.x,
                    (corner & 2u) != 0 ? maximum.y : minimum.y,
                    (corner & 4u) != 0 ? maximum.z : minimum.z,
                };
            }
            std::array<std::optional<ImVec2>, 8> projected{};
            for (std::size_t corner = 0; corner < corners.size(); ++corner) {
                projected[corner] = project(corners[corner]);
            }
            for (const auto &edge : edges) {
                if (projected[edge[0]] && projected[edge[1]]) {
                    // The overlay is a hierarchy debug view and intentionally ignores scene depth.
                    drawList->AddLine(*projected[edge[0]], *projected[edge[1]], color, 1.25f);
                }
            }
        }
    }

    void drawControlPanel() {
        const auto *viewport = ImGui::GetMainViewport();
        constexpr float margin = 12.0f;
        const auto panelWidth = std::max(1.0f, std::min(300.0f, viewport->WorkSize.x - 2.0f * margin));
        const auto panelHeight = std::max(1.0f, std::min(430.0f, viewport->WorkSize.y - 2.0f * margin));
        ImGui::SetNextWindowPos({viewport->WorkPos.x + margin, viewport->WorkPos.y + margin}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({panelWidth, panelHeight}, ImGuiCond_Always);
        constexpr ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                           ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings;
        if (ImGui::Begin("Software BVH", nullptr, flags)) {
            ImGui::Text("Mesh: %u triangles", triangleCount);
            ImGui::Text("BVH: %u nodes, depth %u", static_cast<std::uint32_t>(cpuNodes.size()),
                        static_cast<std::uint32_t>(nodesPerDepth.size() - 1));
            ImGui::SeparatorText("Build");
            constexpr std::array<const char *, 3> strategyNames = {
                "Longest-axis median", "Centroid midpoint", "12-bucket SAH",
            };
            auto strategyIndex = static_cast<int>(selectedStrategy);
            if (ImGui::Combo("Strategy", &strategyIndex, strategyNames.data(),
                             static_cast<int>(strategyNames.size()))) {
                selectedStrategy = static_cast<BvhBuildStrategy>(strategyIndex);
            }
            if (ImGui::Button("Build BVH", {-1.0f, 0.0f})) {
                rebuildBvh();
            }
            ImGui::Text("Active: %s", strategyName(activeStrategy).data());
            ImGui::Text("CPU build: %.3f ms", cpuBuildMilliseconds);
            if (!buildError.empty()) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1.0f, 0.35f, 0.3f, 1.0f});
                ImGui::TextWrapped("Build failed: %s", buildError.c_str());
                ImGui::PopStyleColor();
            }
            ImGui::SeparatorText("Traversal");
            if (statisticsValid) {
                const auto pixelCount = static_cast<double>(swapchainExtent().width) * swapchainExtent().height;
                ImGui::Text("Node tests / ray: %.2f", static_cast<double>(statistics.nodeTests) / pixelCount);
                ImGui::Text("Triangle tests / ray: %.2f",
                            static_cast<double>(statistics.triangleTests) / pixelCount);
            } else {
                ImGui::TextUnformatted("Waiting for GPU statistics...");
            }

            ImGui::SeparatorText("Hierarchy");
            ImGui::Checkbox("Show AABB overlay", &showBounds);
            ImGui::BeginDisabled(!showBounds || maxVisibleDepth == 0);
            ImGui::SliderInt("Level", &selectedDepth, 0, static_cast<int>(maxVisibleDepth));
            ImGui::EndDisabled();
            ImGui::TextUnformatted("Left drag: orbit  Middle drag: pan");
            ImGui::TextUnformatted("Wheel: zoom  R: reset camera");
            if (ImGui::Button("Reset camera", {-1.0f, 0.0f})) {
                camera.reset();
                previousDragCursor.reset();
            }
        }
        ImGui::End();
    }

    void updateCameraFromInput() {
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
                    camera.orbit(horizontalPixels * orbitRadiansPerPixel,
                                 verticalPixels * orbitRadiansPerPixel);
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
        BvhApplication application;
        application.run();
    } catch (const std::exception &error) {
        std::println("Error: {}", error.what());
        return 1;
    }
    return 0;
}
