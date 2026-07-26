add_library(common)

target_sources(common PUBLIC
    FILE_SET cxx_modules TYPE CXX_MODULES FILES
        common.ixx
        debug.ixx
        io.ixx
        window.ixx
        windowed_app.ixx
        vulkan_instance.ixx
)

target_link_libraries(common
    PUBLIC Vulkan::cppm
    PRIVATE
        Vulkan::Headers
        glfw
)
