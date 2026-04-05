get_filename_component(dirname ${CMAKE_CURRENT_SOURCE_DIR} NAME)

file(GLOB srcs *.cpp)

add_executable(${dirname} ${srcs})

target_link_libraries(${dirname} PRIVATE 
Vulkan::cppm 
std_module
common
glfw
glm::glm
)

find_program(GLSLC_EXECUTABLE NAMES glslc HINTS "$ENV{VULKAN_SDK}/Bin" REQUIRED)

set(SHADER_SPV_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(TO_CMAKE_PATH "${SHADER_SPV_OUTPUT_DIR}" SHADER_SPV_OUTPUT_DIR)
target_compile_definitions(${dirname} PRIVATE "SHADER_SPV_DIR=\"${SHADER_SPV_OUTPUT_DIR}\"")

# 编译 shaders 下各源文件为同目录下的 *.spv（如 triangle.vert -> triangle.vert.spv）

file(GLOB shader_sources "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.*")

foreach(shader ${shader_sources})
    get_filename_component(shader_name ${shader} NAME)
    set(output_spv "${SHADER_SPV_OUTPUT_DIR}/${shader_name}.spv")
    add_custom_command(
        OUTPUT ${output_spv}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_SPV_OUTPUT_DIR}"
        COMMAND ${GLSLC_EXECUTABLE} ${shader} -o ${output_spv}
        DEPENDS ${shader}
        COMMENT "Compiling shader ${shader_name} to SPIR-V"
        VERBATIM
    )
    list(APPEND shader_outputs ${output_spv})
endforeach()

set(target_name ${dirname}_compile_shaders)
add_custom_target(
    ${target_name}
    DEPENDS ${shader_outputs}
)
add_dependencies(${dirname} ${target_name})
