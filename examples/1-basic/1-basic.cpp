import vulkan_hpp;
import std;
import common;

class HelloTriangleApplication {
public:
    void run() {
        initVulkan();
    }

private:
    vk::raii::Context context;
    vk::raii::Instance instance = nullptr;
    vk::raii::Device device = nullptr;
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::detail::DispatchLoaderDynamic dldy;

    std::uint32_t graphicsQueueFamilyIndex = 0;

    vk::raii::CommandPool commandPool = nullptr;
    vk::raii::CommandBuffers commandBuffers = nullptr;  // must be destroyed before commandPool

    void initVulkan() {
        createContext();
        createInstance();
        selectPhysicalDevice();
        createDevice();
        createCommandPool();
        createCommandBuffer();
    }
    void createContext() {
        auto instanceLayerProperties = context.enumerateInstanceLayerProperties();
        std::println("Instance supported layers: {}", instanceLayerProperties.size());
        for (auto &e : instanceLayerProperties) {
            std::println("{}", e.layerName.data());
        }
        auto instanceExtensionProperties = context.enumerateInstanceExtensionProperties();
        std::println("Instance supported extensions: {}", instanceExtensionProperties.size());
        for (auto &e : instanceExtensionProperties) {
            std::println("{}", e.extensionName.data());
        }
    }

    void createInstance() {
        // initialize minimal set of function pointers
        dldy.init();

        auto apiVersion = context.enumerateInstanceVersion();

        std::println("API Version: {}.{}.{}.{}", vk::apiVersionMajor(apiVersion), vk::apiVersionMinor(apiVersion),
                     vk::apiVersionPatch(apiVersion), vk::apiVersionVariant(apiVersion));

        vk::ApplicationInfo appInfo{.pApplicationName = "Hello Triangle",
                                    .applicationVersion = vk::makeApiVersion(1, 0, 0, 0),
                                    .pEngineName = "No Engine",
                                    .engineVersion = vk::makeApiVersion(1, 0, 0, 0),
                                    .apiVersion = apiVersion};
        //   .apiVersion = vk::ApiVersion14};

        std::vector<const char *> enabledLayers = {
            "VK_LAYER_KHRONOS_validation",
            "VK_LAYER_LUNARG_crash_diagnostic",
            // "VK_LAYER_LUNARG_api_dump",
        };

        std::vector<const char *> enabledExtensions = {
            "VK_KHR_surface",
            "VK_KHR_win32_surface",
            "VK_EXT_debug_utils",
            "VK_KHR_get_physical_device_properties2",
            "VK_KHR_get_surface_capabilities2",
            "VK_EXT_surface_maintenance1",
        };

        vk::InstanceCreateInfo createInfo{.pApplicationInfo = &appInfo,
                                          .enabledLayerCount = static_cast<std::uint32_t>(enabledLayers.size()),
                                          .ppEnabledLayerNames = enabledLayers.data(),
                                          .enabledExtensionCount = static_cast<std::uint32_t>(enabledExtensions.size()),
                                          .ppEnabledExtensionNames = enabledExtensions.data()};

        instance = vk::raii::Instance(context, createInfo);

        // load function pointers with created instance
        dldy.init(*instance);
    }
    void selectPhysicalDevice() {
        // create a dispatcher, based on additional vkDevice/vkGetDeviceProcAddr
        auto physicalDevices = instance.enumeratePhysicalDevices();

        debug_throw_if(physicalDevices.empty(), "Failed to find GPUs with Vulkan support!");

        for (auto &e : physicalDevices) {
            auto properties = e.getProperties();
            std::println("Physical Device: {}, version:{}.{}.{}, ", properties.deviceName.data(),
                         vk::apiVersionMajor(properties.apiVersion), vk::apiVersionMinor(properties.apiVersion),
                         vk::apiVersionPatch(properties.apiVersion));
        }
        // TODO: select physical device based on properties and features
        physicalDevice = physicalDevices[0];
    }

    void createDevice() {
        auto deviceExtensionProperties = physicalDevice.enumerateDeviceExtensionProperties();
        std::println("Device supported extensions: {}", deviceExtensionProperties.size());
        for (auto &e : deviceExtensionProperties) {
            std::println("{}", e.extensionName.data());
        }

        // get the QueueFamilyProperties of the first PhysicalDevice
        std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

        // get the first index into queueFamiliyProperties which supports graphics
        auto propertyIterator = std::find_if(
            queueFamilyProperties.begin(), queueFamilyProperties.end(),
            [](vk::QueueFamilyProperties const &qfp) { return qfp.queueFlags & vk::QueueFlagBits::eGraphics; });

        graphicsQueueFamilyIndex = std::distance(queueFamilyProperties.begin(), propertyIterator);

        debug_throw_if(graphicsQueueFamilyIndex >= queueFamilyProperties.size(), "Failed to find any queue families!");

        std::vector<float> queuePriorities = {1.0f};
        vk::DeviceQueueCreateInfo queueCreateInfo{
            .queueFamilyIndex = static_cast<std::uint32_t>(graphicsQueueFamilyIndex),
            .queueCount = 1,
            .pQueuePriorities = queuePriorities.data(),
        };

        std::vector<const char*> enabledDeviceExtensions = {
            "VK_KHR_swapchain"
        };

        vk::DeviceCreateInfo deviceCreateInfo{
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queueCreateInfo,
            .enabledExtensionCount = static_cast<std::uint32_t>(enabledDeviceExtensions.size()),
            .ppEnabledExtensionNames = enabledDeviceExtensions.data(),
        };

        device = physicalDevice.createDevice(deviceCreateInfo);

        // OPTIONAL: load function pointers for a specific vk::Device
        dldy.init(*device);
    }

    void createCommandPool() {
        vk::CommandPoolCreateInfo commandPoolCreateInfo{
            .queueFamilyIndex = static_cast<std::uint32_t>(graphicsQueueFamilyIndex),
        };
        commandPool = device.createCommandPool(commandPoolCreateInfo);
    }

    void createCommandBuffer() {
        vk::CommandBufferAllocateInfo commandBufferAllocateInfo{
            .commandPool = *commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        };
        // commandBuffers is a vector of vk::raii::CommandBuffer
        commandBuffers = vk::raii::CommandBuffers(device, commandBufferAllocateInfo);
    }
};

int main() {
    try {
        HelloTriangleApplication app;
        app.run();
    } catch (const std::exception &e) {
        std::println("Error: {}", e.what());
        return 1;
    }

    return 0;
}