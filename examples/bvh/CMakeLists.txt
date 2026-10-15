get_filename_component(dirname ${CMAKE_CURRENT_SOURCE_DIR} NAME)

file(GLOB srcs CONFIGURE_DEPENDS *.cpp)

add_executable(${dirname} ${srcs})

target_link_libraries(${dirname} PRIVATE
    Vulkan::cppm
    common
    glm::glm
    tinyobjloader::tinyobjloader
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

set(shader_outputs)
foreach(shader ${shader_sources})
    get_filename_component(shader_name ${shader} NAME)
    set(output_spv "${SHADER_SPV_OUTPUT_DIR}/${shader_name}.spv")
    add_custom_command(
        OUTPUT ${output_spv}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_SPV_OUTPUT_DIR}"
        COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.1 ${shader} -o ${output_spv}
        COMMAND ${SPIRV_VAL_EXECUTABLE} --target-env vulkan1.1 ${output_spv}
        DEPENDS ${shader}
        COMMENT "Compiling and validating BVH shader ${shader_name}"
        VERBATIM
    )
    list(APPEND shader_outputs ${output_spv})
endforeach()

add_custom_target(${dirname}_compile_shaders DEPENDS ${shader_outputs})
add_dependencies(${dirname} ${dirname}_compile_shaders)

set(BUNNY_SOURCE_PATH "${PROJECT_SOURCE_DIR}/assets/models/bunny.obj")
set(BUNNY_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/assets/models")
set(BUNNY_OUTPUT_PATH "${BUNNY_OUTPUT_DIR}/bunny.obj")

add_custom_command(
    OUTPUT "${BUNNY_OUTPUT_PATH}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNNY_OUTPUT_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${BUNNY_SOURCE_PATH}" "${BUNNY_OUTPUT_PATH}"
    DEPENDS "${BUNNY_SOURCE_PATH}"
    COMMENT "Copying bunny.obj for the BVH example"
    VERBATIM
)

add_custom_target(${dirname}_copy_assets DEPENDS "${BUNNY_OUTPUT_PATH}")
add_dependencies(${dirname} ${dirname}_copy_assets)

file(TO_CMAKE_PATH "${BUNNY_OUTPUT_PATH}" BUNNY_OUTPUT_PATH)
target_compile_definitions(${dirname} PRIVATE "BUNNY_OBJ_PATH=\"${BUNNY_OUTPUT_PATH}\"")
