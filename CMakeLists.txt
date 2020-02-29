cmake_minimum_required(VERSION 3.0)
project(opengl_viz_intro LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)


# CUDA
find_package(CUDA REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION} at ${CUDA_TOOLKIT_ROOT_DIR}")
include_directories("${CUDA_INCLUDE_DIRS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Eigen: For linear algebra
#find_package(Eigen3 REQUIRED NO_MODULE)
include_directories(libs/eigen)

# glfw: For windows and displaying
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
include_directories(${GLFW_INCLUDE_DIRS})

# stb: For loading images
include_directories(libs/stb/include)
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

# CUDA code
file(GLOB_RECURSE CUDA_SOURCE_FILES src/*.cu)
add_library(cuda_code ${CUDA_SOURCE_FILES})
set_target_properties(cuda_code PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


set(CXX_LIBRARIES
    #Eigen3::Eigen
    stb
    glad
    imgui
    GL
    stdc++fs
    ${GLFW_STATIC_LIBRARIES}
    ${CUDA_LIBRARIES}
    cuda_code)

add_executable(opengl_viz ${CXX_SOURCE_FILES})
target_link_libraries(opengl_viz ${CXX_LIBRARIES})
