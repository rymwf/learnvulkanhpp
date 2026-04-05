get_filename_component(dirname ${CMAKE_CURRENT_SOURCE_DIR} NAME)

file(GLOB srcs *.cpp)

add_executable(${dirname} ${srcs})

target_link_libraries(${dirname} PRIVATE 
Vulkan::cppm 
std_module
common
)