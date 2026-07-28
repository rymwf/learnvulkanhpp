add_library(common)

target_sources(common PUBLIC
    FILE_SET cxx_modules TYPE CXX_MODULES FILES
        common.ixx
        debug.ixx
        io.ixx
        orbit_camera.ixx
        window.ixx
        windowed_app.ixx
        vulkan_instance.ixx
)

target_link_libraries(common
    PUBLIC
        Vulkan::cppm
        glm::glm
    PRIVATE
        Vulkan::Headers
        glfw
)

set(imgui_package_folder "")
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(package_folder_variable "imgui_PACKAGE_FOLDER_${config}")
    if(DEFINED ${package_folder_variable})
        set(imgui_package_folder "${${package_folder_variable}}")
        break()
    endif()
endforeach()

if(NOT imgui_package_folder)
    message(FATAL_ERROR "The ImGui package folder is unavailable")
endif()

set(imgui_bindings_dir "${imgui_package_folder}/res/bindings")
foreach(binding_file imgui_impl_glfw.cpp imgui_impl_glfw.h imgui_impl_vulkan.cpp imgui_impl_vulkan.h)
    if(NOT EXISTS "${imgui_bindings_dir}/${binding_file}")
        message(FATAL_ERROR "The ImGui package does not contain res/bindings/${binding_file}")
    endif()
endforeach()

add_library(common_imgui)

target_sources(common_imgui
    PRIVATE
        "${imgui_bindings_dir}/imgui_impl_glfw.cpp"
        "${imgui_bindings_dir}/imgui_impl_vulkan.cpp"
    PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
            imgui_overlay.ixx
)

target_include_directories(common_imgui PRIVATE "${imgui_bindings_dir}")

target_link_libraries(common_imgui
    PUBLIC
        Vulkan::cppm
        common
        imgui::imgui
    PRIVATE
        Vulkan::Headers
        Vulkan::Vulkan
        glfw
)
