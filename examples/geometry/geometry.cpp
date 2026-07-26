import vulkan;
import std;
import common;

#ifndef SHADER_SPV_DIR
#error "SHADER_SPV_DIR must be set by CMake (see examples/geometry/CMakeLists.txt)"
#endif

class GeometryShaderApplication : public WindowedVulkanApp {
public:
    GeometryShaderApplication()
        : WindowedVulkanApp({
              .window = {
                  .width = 1024,
                  .height = 768,
                  .title = "Vulkan Geometry Shader",
              },
              .apiVersion = vk::ApiVersion10,
          }) {}

private:
    vk::raii::ShaderModule vertShaderModule = nullptr;
    vk::raii::ShaderModule geometryShaderModule = nullptr;
    vk::raii::ShaderModule fragShaderModule = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;

    bool supportsDeviceFeatures(const vk::raii::PhysicalDevice &candidate) const override {
        return candidate.getFeatures().geometryShader;
    }

    vk::PhysicalDeviceFeatures enabledDeviceFeatures() const override {
        return {
            .geometryShader = true,
        };
    }

    void createApplicationResources() override {
        createGraphicsPipeline();
    }

    void recordFrameCommands(vk::raii::CommandBuffer &commandBuffer, std::uint32_t imageIndex) override {
        beginSwapchainRenderPass(commandBuffer, imageIndex);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
        commandBuffer.draw(3, 1, 0, 0);
        commandBuffer.endRenderPass();
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
        vertShaderModule = device().createShaderModule(vertModuleInfo);
        geometryShaderModule = device().createShaderModule(geometryModuleInfo);
        fragShaderModule = device().createShaderModule(fragModuleInfo);

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

        const auto extent = swapchainExtent();
        vk::Viewport viewport{
            .x = 0.f,
            .y = 0.f,
            .width = static_cast<float>(extent.width),
            .height = static_cast<float>(extent.height),
            .minDepth = 0.f,
            .maxDepth = 1.f,
        };
        vk::Rect2D scissor{.offset = {0, 0}, .extent = extent};
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
        pipelineLayout = device().createPipelineLayout(pipelineLayoutInfo);

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
            .renderPass = swapchainRenderPass(),
            .subpass = 0,
        };
        graphicsPipeline = device().createGraphicsPipeline(nullptr, pipelineInfo);
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
