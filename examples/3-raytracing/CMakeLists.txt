get_filename_component(dirname ${CMAKE_CURRENT_SOURCE_DIR} NAME)

file(GLOB srcs CONFIGURE_DEPENDS *.cpp)

add_executable(${dirname} ${srcs})

target_link_libraries(${dirname} PRIVATE
    Vulkan::cppm
    common
)

find_program(GLSLC_EXECUTABLE NAMES glslc HINTS "$ENV{VULKAN_SDK}/Bin" REQUIRED)
find_program(SPIRV_VAL_EXECUTABLE NAMES spirv-val HINTS "$ENV{VULKAN_SDK}/Bin" REQUIRED)

set(SHADER_SPV_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(TO_CMAKE_PATH "${SHADER_SPV_OUTPUT_DIR}" SHADER_SPV_OUTPUT_DIR)
target_compile_definitions(${dirname} PRIVATE "SHADER_SPV_DIR=\"${SHADER_SPV_OUTPUT_DIR}\"")

file(GLOB shader_sources CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.rgen"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.rmiss"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.rchit"
)

set(shader_outputs)
foreach(shader ${shader_sources})
    get_filename_component(shader_name ${shader} NAME)
    set(output_spv "${SHADER_SPV_OUTPUT_DIR}/${shader_name}.spv")
    add_custom_command(
        OUTPUT ${output_spv}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_SPV_OUTPUT_DIR}"
        COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.2 ${shader} -o ${output_spv}
        COMMAND ${SPIRV_VAL_EXECUTABLE} --target-env vulkan1.2 ${output_spv}
        DEPENDS ${shader}
        COMMENT "Compiling and validating ray tracing shader ${shader_name}"
        VERBATIM
    )
    list(APPEND shader_outputs ${output_spv})
endforeach()

set(shader_target ${dirname}_compile_shaders)
add_custom_target(${shader_target} DEPENDS ${shader_outputs})
add_dependencies(${dirname} ${shader_target})
