cmake_minimum_required(VERSION 3.22)
project(First CUDA)

set(BOOST_ROOT "E:/Paul/C++/boost_1_79_0/boost_1_79_0") # either set it here or from the command line
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(Boost REQUIRED)

link_directories("E:/Paul/C++/boost_1_79_0/boost_1_79_0/stage/lib")
include_directories(${Boost_INCLUDE_DIRS})
include_directories("E:/Paul/C++/Commons")
include_directories("E:/Paul/Cuda/Commons")

set(CMAKE_CUDA_STANDARD 14)

add_executable(First main.cu)

target_link_libraries(First ${Boost_LIBRARIES})

set_target_properties(First PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
