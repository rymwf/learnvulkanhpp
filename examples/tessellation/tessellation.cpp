import vulkan;
import std;
import common;

#ifndef SHADER_SPV_DIR
#error "SHADER_SPV_DIR must be set by CMake (see examples/tessellation/CMakeLists.txt)"
#endif

struct TessellationPushConstants {
    float elapsedSeconds;
    float aspectRatio;
};

static_assert(sizeof(TessellationPushConstants) == 2 * sizeof(float));

class TessellationShaderApplication : public WindowedVulkanApp {
public:
    TessellationShaderApplication()
        : WindowedVulkanApp({
              .window = {
                  .width = 1024,
                  .height = 768,
                  .title = "Vulkan Tessellation Shader",
              },
              .apiVersion = vk::ApiVersion11,
          }) {}

private:
    vk::raii::ShaderModule vertShaderModule = nullptr;
    vk::raii::ShaderModule tessControlShaderModule = nullptr;
    vk::raii::ShaderModule tessEvaluationShaderModule = nullptr;
    vk::raii::ShaderModule fragShaderModule = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;

    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    bool supportsDeviceFeatures(const vk::raii::PhysicalDevice &candidate) const override {
        const auto features = candidate.getFeatures();
        return features.tessellationShader && features.fillModeNonSolid;
    }

    vk::PhysicalDeviceFeatures enabledDeviceFeatures() const override {
        return {
            .tessellationShader = true,
            .fillModeNonSolid = true,
        };
    }

    void createApplicationResources() override {
        createGraphicsPipeline();
    }

    void recordFrameCommands(vk::raii::CommandBuffer &commandBuffer, std::uint32_t imageIndex) override {
        const auto extent = swapchainExtent();
        TessellationPushConstants pushConstants{
            .elapsedSeconds =
                std::chrono::duration<float>(std::chrono::steady_clock::now() - startTime).count(),
            .aspectRatio = static_cast<float>(extent.width) / static_cast<float>(extent.height),
        };

        beginSwapchainRenderPass(commandBuffer, imageIndex);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
        commandBuffer.pushConstants(*pipelineLayout, vk::ShaderStageFlagBits::eTessellationEvaluation, 0,
                                    static_cast<std::uint32_t>(sizeof(pushConstants)), &pushConstants);
        commandBuffer.draw(4, 1, 0, 0);
        commandBuffer.endRenderPass();
    }

    void createGraphicsPipeline() {
        const std::filesystem::path shaderDir{SHADER_SPV_DIR};
        const std::vector<std::uint32_t> vertSpirv = readSpirvFile(shaderDir / "tessellation.vert.spv");
        const std::vector<std::uint32_t> tessControlSpirv =
            readSpirvFile(shaderDir / "tessellation.tesc.spv");
        const std::vector<std::uint32_t> tessEvaluationSpirv =
            readSpirvFile(shaderDir / "tessellation.tese.spv");
        const std::vector<std::uint32_t> fragSpirv = readSpirvFile(shaderDir / "tessellation.frag.spv");

        vk::ShaderModuleCreateInfo vertModuleInfo{
            .codeSize = vertSpirv.size() * sizeof(std::uint32_t),
            .pCode = vertSpirv.data(),
        };
        vk::ShaderModuleCreateInfo tessControlModuleInfo{
            .codeSize = tessControlSpirv.size() * sizeof(std::uint32_t),
            .pCode = tessControlSpirv.data(),
        };
        vk::ShaderModuleCreateInfo tessEvaluationModuleInfo{
            .codeSize = tessEvaluationSpirv.size() * sizeof(std::uint32_t),
            .pCode = tessEvaluationSpirv.data(),
        };
        vk::ShaderModuleCreateInfo fragModuleInfo{
            .codeSize = fragSpirv.size() * sizeof(std::uint32_t),
            .pCode = fragSpirv.data(),
        };
        vertShaderModule = device().createShaderModule(vertModuleInfo);
        tessControlShaderModule = device().createShaderModule(tessControlModuleInfo);
        tessEvaluationShaderModule = device().createShaderModule(tessEvaluationModuleInfo);
        fragShaderModule = device().createShaderModule(fragModuleInfo);

        vk::PipelineShaderStageCreateInfo vertStage{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = *vertShaderModule,
            .pName = "main",
        };
        vk::PipelineShaderStageCreateInfo tessControlStage{
            .stage = vk::ShaderStageFlagBits::eTessellationControl,
            .module = *tessControlShaderModule,
            .pName = "main",
        };
        vk::PipelineShaderStageCreateInfo tessEvaluationStage{
            .stage = vk::ShaderStageFlagBits::eTessellationEvaluation,
            .module = *tessEvaluationShaderModule,
            .pName = "main",
        };
        vk::PipelineShaderStageCreateInfo fragStage{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = *fragShaderModule,
            .pName = "main",
        };
        std::array<vk::PipelineShaderStageCreateInfo, 4> shaderStages = {
            vertStage,
            tessControlStage,
            tessEvaluationStage,
            fragStage,
        };

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::ePatchList,
        };
        vk::PipelineTessellationStateCreateInfo tessellationState{
            .patchControlPoints = 4,
        };

        const auto extent = swapchainExtent();
        vk::Viewport viewport{
            .x = 0.f,
            .y = static_cast<float>(extent.height),
            .width = static_cast<float>(extent.width),
            .height = -static_cast<float>(extent.height),
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
            .polygonMode = vk::PolygonMode::eLine,
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

        vk::PushConstantRange pushConstantRange{
            .stageFlags = vk::ShaderStageFlagBits::eTessellationEvaluation,
            .offset = 0,
            .size = static_cast<std::uint32_t>(sizeof(TessellationPushConstants)),
        };
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstantRange,
        };
        pipelineLayout = device().createPipelineLayout(pipelineLayoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo{
            .stageCount = static_cast<std::uint32_t>(shaderStages.size()),
            .pStages = shaderStages.data(),
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pTessellationState = &tessellationState,
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
        TessellationShaderApplication app;
        app.run();
    } catch (const std::exception &e) {
        std::println("Error: {}", e.what());
        return 1;
    }

    return 0;
}
