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
        imgui_overlay.ixx
)

target_link_libraries(common
    PUBLIC
        Vulkan::cppm
        glm::glm
        imgui::imgui
    PRIVATE
        Vulkan::Headers
        glfw
)