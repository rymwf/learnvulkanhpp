add_subdirectory(glfw)

add_library(glm_module)
add_library(glm::glm ALIAS glm_module)

target_include_directories(glm_module PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_sources(glm_module
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/glm/glm
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/glm/glm/glm.cppm
)