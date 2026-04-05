file(GLOB srcs *.cpp)

foreach(example ${srcs}) 

get_filename_component(aa ${example} NAME_WLE)

add_executable(${aa} ${example})

target_include_directories(${aa} PRIVATE 
${CMAKE_SOURCE_DIR}
${CMAKE_SOURCE_DIR}/3rdparty/include
)

target_link_libraries(${aa} PRIVATE 
Vulkan::cppm 
glm::glm
std_module
common
glfw
)

endforeach()