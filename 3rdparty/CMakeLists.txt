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

#add std module
add_library(std_module)

target_sources(std_module
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES
        BASE_DIRS $ENV{UCRT64_PATH}/share/libc++/v1
        FILES $ENV{UCRT64_PATH}/share/libc++/v1/std.cppm
)
target_compile_options(std_module PUBLIC -stdlib=libc++)
target_link_libraries(std_module PUBLIC libc++)