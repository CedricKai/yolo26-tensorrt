cmake_minimum_required(VERSION 3.16)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin /usr/bin/g++-9")
if(UNIX)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")
    message(STATUS ">>> Build on Ubuntu Linux")
endif ()

project(yolo26 LANGUAGES CXX CUDA)

# C++ / CUDA 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    add_definitions(-DAPI_EXPORTS)
    add_definitions(-DNOMINMAX)
endif()


if(UNIX)
    set(CUDA_ROOT "/usr/local/cuda-12.6")
    set(TENSORRT_ROOT "/home/cedric/TensorRT-10.7.0.23")
    link_directories(${CUDA_ROOT}/lib64)
    link_directories(${TENSORRT_ROOT}/lib)
    find_package(OpenCV REQUIRED
            PATHS /usr/local
            NO_DEFAULT_PATH
    )
    message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV Path: ${OpenCV_DIR}")
endif()

if(WIN32)
    set(CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
    set(TENSORRT_ROOT "S:/TensorRT-8.6.1.6")
    set(OPENCV_ROOT "S:/opencv/build")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(OPENCV_LIB ${OPENCV_ROOT}/x64/vc16/lib/opencv_world481d.lib)
    else()
        set(OPENCV_LIB ${OPENCV_ROOT}/x64/vc16/lib/opencv_world481.lib)
    endif()
endif()

# 公共头文件（所有目标通用）
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/plugin)

# 平台相关头文件
if(UNIX)
    include_directories(${CUDA_ROOT}/include)
    include_directories(${TENSORRT_ROOT}/include)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()
if(WIN32)
    include_directories(${CUDA_ROOT}/include)
    include_directories(${TENSORRT_ROOT}/include)
    include_directories(${OPENCV_ROOT}/include)
endif()


# TRT yololayer 插件库
add_library(trt_plugins SHARED ${PROJECT_SOURCE_DIR}/plugin/yololayer.cu)
target_link_libraries(trt_plugins PRIVATE nvinfer cudart)

# 收集源码
file(GLOB_RECURSE SRCS
        ${PROJECT_SOURCE_DIR}/src/*.cpp
        ${PROJECT_SOURCE_DIR}/src/*.cu
)

# 主程序
add_executable(yolo26_det
        ${PROJECT_SOURCE_DIR}/yolo26_det.cpp
        ${SRCS}
)

# 平台链接
if(UNIX)
    target_link_libraries(yolo26_det PRIVATE
            trt_plugins
            nvinfer
            cudart
            ${OpenCV_LIBS}
    )
endif()

if(WIN32)
    target_link_libraries(yolo26_det PRIVATE
            trt_plugins
            nvinfer
            cudart
            ${OPENCV_LIB}
    )
endif()
