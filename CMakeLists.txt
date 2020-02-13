cmake_minimum_required(VERSION 3.15)
project(opengl_viz_intro LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

# Eigen: For linear algebra
find_package(Eigen3 REQUIRED NO_MODULE)

# glfw: For windows and displaying
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
include_directories(${GLFW_INCLUDE_DIRS})

# stb: For loading images
include_directories(libs/stb)
set(STB_SOURCE_FILES
    libs/stb/src/stb_image.cpp)
add_library(stb ${STB_SOURCE_FILES})

# glad: For loading OpenGL driver
include_directories(libs/glad/include)
add_library(glad STATIC libs/glad/src/glad.c)

# imgui: GUI Library
include_directories(libs/imgui)
file(GLOB_RECURSE IMGUI_SOURCE_FILES libs/imgui/*.cpp)
add_library(imgui ${IMGUI_SOURCE_FILES})

# C++ code
include_directories(include)
file(GLOB_RECURSE CXX_SOURCE_FILES src/*.cpp)

set(CXX_LIBRARIES
    Eigen3::Eigen
    stb
    glad
    imgui
    ${GLFW_STATIC_LIBRARIES})

add_executable(opengl_viz ${CXX_SOURCE_FILES})
target_link_libraries(opengl_viz ${CXX_LIBRARIES})
