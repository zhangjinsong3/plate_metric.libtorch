cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(plate_metric)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(OpenCV_DIR /home/zjs/opt/OpenCV-3.4.0-cuda-9.0/share/OpenCV)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(Torch_DIR /home/zjs/opt/libtorch/share/cmake/Torch)
set(Caffe2_DIR /home/zjs/opt/libtorch/share/cmake/Caffe2)
find_package(Torch REQUIRED)

add_executable(metric metric.cpp)
target_link_libraries(metric ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET metric PROPERTY CXX_STANDARD 11)

