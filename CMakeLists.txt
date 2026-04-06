cmake_minimum_required(VERSION 4.1 FATAL_ERROR)

get_filename_component(MY_PROJECT_NAME ${CMAKE_SOURCE_DIR} NAME_WLE)
project(${MY_PROJECT_NAME} VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_EXTENSIONS off)
set(CMAKE_CXX_SCAN_FOR_MODULES on)
set(CMAKE_CXX_MODULE_STD on)

# Vulkan
find_package (Vulkan REQUIRED)

add_library(VulkanCppModule)
add_library(Vulkan::cppm ALIAS VulkanCppModule)

target_compile_definitions(VulkanCppModule PUBLIC
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
    VULKAN_HPP_NO_CONSTRUCTORS #use designated initializers 
)
if(WIN32)
    target_compile_definitions(VulkanCppModule PUBLIC
        VK_USE_PLATFORM_WIN32_KHR
    )
endif()

target_include_directories(VulkanCppModule PRIVATE "${Vulkan_INCLUDE_DIR}")

# use dynamic dispatcher, so that we don't have to link against the static library, and can load the functions at runtime
# target_link_libraries(VulkanCppModule PUBLIC Vulkan::Vulkan)

set_target_properties(VulkanCppModule PROPERTIES CXX_STANDARD 23)

target_sources(VulkanCppModule
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES
        BASE_DIRS "${Vulkan_INCLUDE_DIR}"
        FILES "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm"
)

add_subdirectory(3rdparty)
add_subdirectory(common)
add_subdirectory(examples)