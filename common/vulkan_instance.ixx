export module common.vulkan_instance;

import std;
import vulkan;
import common.debug;

namespace {
bool hasLayer(const std::vector<vk::LayerProperties> &properties, std::string_view name) {
    for (const auto &property : properties) {
        if (name == property.layerName.data()) {
            return true;
        }
    }
    return false;
}

bool hasExtension(const std::vector<vk::ExtensionProperties> &properties, std::string_view name) {
    for (const auto &property : properties) {
        if (name == property.extensionName.data()) {
            return true;
        }
    }
    return false;
}

bool contains(const std::vector<std::string> &values, std::string_view value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

vk::Bool32 validationCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT,
                              vk::DebugUtilsMessageTypeFlagsEXT,
                              const vk::DebugUtilsMessengerCallbackDataEXT *callbackData,
                              void *) {
    if (callbackData != nullptr && callbackData->pMessage != nullptr) {
        std::println("[validation] {}", callbackData->pMessage);
    }
    return 0;
}
}  // namespace

export struct VulkanInstanceConfig {
    std::string applicationName;
    std::uint32_t applicationVersion = vk::makeApiVersion(0, 1, 0, 0);
    std::uint32_t apiVersion = vk::ApiVersion10;
    std::vector<std::string> requiredExtensions;
    bool enableValidation = debug_mode;
};

export class VulkanInstance {
public:
    explicit VulkanInstance(VulkanInstanceConfig config) {
        if (config.applicationName.empty()) {
            throw std::runtime_error("Vulkan application name must not be empty");
        }

        const auto loaderApiVersion = context_.enumerateInstanceVersion();
        if (loaderApiVersion < config.apiVersion) {
            throw std::runtime_error(std::format("Vulkan {}.{} is required, but the loader supports {}.{}",
                                                 vk::apiVersionMajor(config.apiVersion),
                                                 vk::apiVersionMinor(config.apiVersion),
                                                 vk::apiVersionMajor(loaderApiVersion),
                                                 vk::apiVersionMinor(loaderApiVersion)));
        }

        const auto availableLayers = context_.enumerateInstanceLayerProperties();
        const auto availableExtensions = context_.enumerateInstanceExtensionProperties();

        std::vector<const char *> enabledLayers;
        std::vector<std::string> enabledExtensions = std::move(config.requiredExtensions);
        bool validationEnabled = false;
        bool debugUtilsEnabled = false;

        if (config.enableValidation) {
            if (hasLayer(availableLayers, "VK_LAYER_KHRONOS_validation")) {
                enabledLayers.push_back("VK_LAYER_KHRONOS_validation");
                validationEnabled = true;
                if (hasExtension(availableExtensions, "VK_EXT_debug_utils")) {
                    if (!contains(enabledExtensions, "VK_EXT_debug_utils")) {
                        enabledExtensions.emplace_back("VK_EXT_debug_utils");
                    }
                    debugUtilsEnabled = true;
                } else {
                    std::println("VK_EXT_debug_utils is unavailable; validation messages will not use a debug messenger");
                }
            } else {
                std::println("VK_LAYER_KHRONOS_validation is unavailable; continuing without validation");
            }
        }

        for (const auto &extension : enabledExtensions) {
            if (!hasExtension(availableExtensions, extension)) {
                throw std::runtime_error(std::format("Required instance extension is unavailable: {}", extension));
            }
        }

        std::vector<const char *> extensionNames;
        extensionNames.reserve(enabledExtensions.size());
        for (const auto &extension : enabledExtensions) {
            extensionNames.push_back(extension.c_str());
        }

        vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo{
            .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                               vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
            .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                           vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                           vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
            .pfnUserCallback = validationCallback,
        };
        vk::ApplicationInfo applicationInfo{
            .pApplicationName = config.applicationName.c_str(),
            .applicationVersion = config.applicationVersion,
            .pEngineName = "No Engine",
            .engineVersion = vk::makeApiVersion(0, 1, 0, 0),
            .apiVersion = config.apiVersion,
        };
        vk::InstanceCreateInfo createInfo{
            .pNext = debugUtilsEnabled ? &debugCreateInfo : nullptr,
            .pApplicationInfo = &applicationInfo,
            .enabledLayerCount = static_cast<std::uint32_t>(enabledLayers.size()),
            .ppEnabledLayerNames = enabledLayers.data(),
            .enabledExtensionCount = static_cast<std::uint32_t>(extensionNames.size()),
            .ppEnabledExtensionNames = extensionNames.data(),
        };

        instance_ = vk::raii::Instance(context_, createInfo);
        if (debugUtilsEnabled) {
            debugMessenger_ = vk::raii::DebugUtilsMessengerEXT(instance_, debugCreateInfo);
        }

        std::println("Vulkan instance: API {}.{}.{}{}", vk::apiVersionMajor(config.apiVersion),
                     vk::apiVersionMinor(config.apiVersion), vk::apiVersionPatch(config.apiVersion),
                     validationEnabled ? " with validation" : "");
    }

    VulkanInstance(const VulkanInstance &) = delete;
    VulkanInstance &operator=(const VulkanInstance &) = delete;
    VulkanInstance(VulkanInstance &&) = delete;
    VulkanInstance &operator=(VulkanInstance &&) = delete;

    [[nodiscard]] const vk::raii::Instance &instance() const noexcept {
        return instance_;
    }

private:
    vk::raii::Context context_;
    vk::raii::Instance instance_ = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger_ = nullptr;
};
