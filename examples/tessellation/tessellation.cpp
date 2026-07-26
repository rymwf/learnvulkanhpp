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
              .apiVersion = vk::ApiVersion10,
          }) {}

private:
    std::vector<vk::raii::ImageView> imageViews;
    std::vector<vk::raii::DeviceMemory> depthImageMemories;
    std::vector<vk::raii::Image> depthImages;
    std::vector<vk::raii::ImageView> depthImageViews;

    vk::raii::RenderPass renderPass = nullptr;
    vk::raii::ShaderModule vertShaderModule = nullptr;
    vk::raii::ShaderModule tessControlShaderModule = nullptr;
    vk::raii::ShaderModule tessEvaluationShaderModule = nullptr;
    vk::raii::ShaderModule fragShaderModule = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;

    std::vector<vk::raii::Framebuffer> framebuffers;
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

        const auto extent = swapchainExtent();
        TessellationPushConstants pushConstants{
            .elapsedSeconds =
                std::chrono::duration<float>(std::chrono::steady_clock::now() - startTime).count(),
            .aspectRatio = static_cast<float>(extent.width) / static_cast<float>(extent.height),
        };

        commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
        commandBuffer.pushConstants(*pipelineLayout, vk::ShaderStageFlagBits::eTessellationEvaluation, 0,
                                    static_cast<std::uint32_t>(sizeof(pushConstants)), &pushConstants);
        commandBuffer.draw(4, 1, 0, 0);
        commandBuffer.endRenderPass();
    }

    void createRenderTargets() {
        const auto extent = swapchainExtent();
        const auto surfaceFormat = swapchainSurfaceFormat();
        for (const auto image : swapchainImages()) {
            vk::ImageViewCreateInfo imageViewCreateInfo{
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
            };
            imageViews.push_back(device().createImageView(imageViewCreateInfo));

            vk::ImageCreateInfo depthImageCreateInfo{
                .imageType = vk::ImageType::e2D,
                .format = vk::Format::eD32Sfloat,
                .extent = {extent.width, extent.height, 1},
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = vk::SampleCountFlagBits::e1,
                .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
            };
            depthImages.push_back(device().createImage(depthImageCreateInfo));
            const vk::MemoryRequirements memoryRequirements = depthImages.back().getMemoryRequirements();

            const auto memoryTypeIndex =
                findMemoryTypeIndex(memoryRequirements, vk::MemoryPropertyFlagBits::eDeviceLocal);

            vk::MemoryAllocateInfo memoryAllocateInfo{
                .allocationSize = memoryRequirements.size,
                .memoryTypeIndex = memoryTypeIndex,
            };
            depthImageMemories.push_back(vk::raii::DeviceMemory(device(), memoryAllocateInfo));
            depthImages.back().bindMemory(*depthImageMemories.back(), 0);

            vk::ImageViewCreateInfo depthImageViewCreateInfo{
                .image = depthImages.back(),
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
            };
            depthImageViews.push_back(device().createImageView(depthImageViewCreateInfo));
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

        vk::AttachmentReference colorRef{
            .attachment = 0,
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
        };
        vk::AttachmentReference depthRef{
            .attachment = 1,
            .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        };

        vk::SubpassDescription subpass{
            .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorRef,
            .pDepthStencilAttachment = &depthRef,
        };

        vk::SubpassDependency dependency{
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

        vk::RenderPassCreateInfo renderPassCreateInfo{
            .attachmentCount = static_cast<std::uint32_t>(attachments.size()),
            .pAttachments = attachments.data(),
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 1,
            .pDependencies = &dependency,
        };
        renderPass = device().createRenderPass(renderPassCreateInfo);
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
            .renderPass = *renderPass,
            .subpass = 0,
        };
        graphicsPipeline = device().createGraphicsPipeline(nullptr, pipelineInfo);
    }

    void createFramebuffers() {
        const auto extent = swapchainExtent();
        for (std::size_t i = 0; i < swapchainImages().size(); ++i) {
            std::array<vk::ImageView, 2> attachments = {*imageViews[i], *depthImageViews[i]};
            vk::FramebufferCreateInfo framebufferCreateInfo{
                .renderPass = *renderPass,
                .attachmentCount = static_cast<std::uint32_t>(attachments.size()),
                .pAttachments = attachments.data(),
                .width = extent.width,
                .height = extent.height,
                .layers = 1,
            };
            framebuffers.push_back(device().createFramebuffer(framebufferCreateInfo));
        }
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
