add_library(common)

file(GLOB common_sources
"${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/*.ixx")

target_sources(common PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES ${common_sources})

# target_link_libraries(common PUBLIC std_module)