cmake_minimum_required(VERSION 4.1 FATAL_ERROR)

# Required before compiler detection when the project uses import std.
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b")

get_filename_component(MY_PROJECT_NAME ${CMAKE_SOURCE_DIR} NAME_WLE)
project(${MY_PROJECT_NAME} VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_EXTENSIONS off)
set(CMAKE_CXX_SCAN_FOR_MODULES on)
set(CMAKE_CXX_MODULE_STD on)

find_package(Vulkan REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(spirv-reflect-static CONFIG REQUIRED)

add_subdirectory(3rdparty)
add_subdirectory(common)
add_subdirectory(examples)
