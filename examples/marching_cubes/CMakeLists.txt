get_filename_component(dirname ${CMAKE_CURRENT_SOURCE_DIR} NAME)

file(GLOB srcs CONFIGURE_DEPENDS *.cpp)

add_executable(${dirname} ${srcs})

target_link_libraries(${dirname} PRIVATE
    Vulkan::cppm
    common
    common_imgui
    glm::glm
)

find_program(GLSLC_EXECUTABLE NAMES glslc HINTS "$ENV{VULKAN_SDK}/Bin" REQUIRED)
find_program(SPIRV_VAL_EXECUTABLE NAMES spirv-val HINTS "$ENV{VULKAN_SDK}/Bin" REQUIRED)

set(SHADER_SPV_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(TO_CMAKE_PATH "${SHADER_SPV_OUTPUT_DIR}" SHADER_SPV_OUTPUT_DIR)
target_compile_definitions(${dirname} PRIVATE "SHADER_SPV_DIR=\"${SHADER_SPV_OUTPUT_DIR}\"")

file(GLOB shader_sources CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.frag"
)
file(GLOB shader_includes CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.glsl")

set(shader_outputs)
foreach(shader ${shader_sources})
    get_filename_component(shader_name ${shader} NAME)
    set(output_spv "${SHADER_SPV_OUTPUT_DIR}/${shader_name}.spv")
    add_custom_command(
        OUTPUT ${output_spv}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_SPV_OUTPUT_DIR}"
        COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.1
                -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders" ${shader} -o ${output_spv}
        COMMAND ${SPIRV_VAL_EXECUTABLE} --target-env vulkan1.1 ${output_spv}
        DEPENDS ${shader} ${shader_includes}
        COMMENT "Compiling and validating Marching Cubes shader ${shader_name}"
        VERBATIM
    )
    list(APPEND shader_outputs ${output_spv})
endforeach()

add_custom_target(${dirname}_compile_shaders DEPENDS ${shader_outputs})
add_dependencies(${dirname} ${dirname}_compile_shaders)
