# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(yolov5_evs)

# Compile options
add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

if (NOT DEFINED ENV{INSTALL_DIR})
    MESSAGE(FATAL_ERROR "Not Defined INSTALL_DIR")
endif()

if (NOT DEFINED ENV{THIRDPART_PATH})
    message(FATAL_ERROR "Not Defined THIRDPART_PATH")
endif()

if (NOT DEFINED ENV{CPU_ARCH})
    message(FATAL_ERROR "Not Defined CPU_ARCH")
endif()

if($ENV{CPU_ARCH} MATCHES "aarch64")
    message("arm architecture detected")
    if(EXISTS "$ENV{INSTALL_DIR}/driver/libmedia_mini.so")
        message("target aarch64 200dk")
    else()
        message("target x86 300")
    endif()
else()
    message("x86 architecture detected")
    message("target x86 300")
endif()

# Header path
#include_directories(
#    $ENV{HOME}/ascend_ddk/${ARCH}/include/
#    $ENV{HOME}/ascend_ddk/${ARCH}/include/opencv4
#    ${INC_PATH}/acllib/include/
#    ../inc/
#)

include_directories(
    $ENV{INSTALL_DIR}/runtime/include/
	$ENV{THIRDPART_PATH}/include/
    ../inc/
)

if(target STREQUAL "Simulator_Function")
    add_compile_options(-DFUNC_SIM)
endif()

# add host lib path
link_directories(
    $ENV{INSTALL_DIR}/runtime/lib64/stub
    $ENV{THIRDPART_PATH}/lib/
)

# add host lib path
#link_directories(
#    /usr/local/lib
#    ${LIB_PATH}
#    $ENV{HOME}/ascend_ddk/${ARCH}/lib/
#    $ENV{HOME}/Ascend/
#    ${INC_PATH}/atc/lib64
#)

add_executable(main
        utils.cpp
        model_process.cpp
        do_process.cpp
        main.cpp)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

if(target STREQUAL "Simulator_Function")
    target_link_libraries(main funcsim)
else()
    target_link_libraries(main ascendcl acl_dvpp stdc++ z jpeg png16 tiff lzma jbig opencv_core opencv_imgproc opencv_imgcodecs ${OpenCV_LIBS} OpenMP::OpenMP_CXX dl rt)
    # target_link_libraries(main ascendcl acl_dvpp stdc++ z jpeg png16 tiff lzma jbig opencv_core opencv_imgproc opencv_imgcodecs ${OpenCV_LIBS} dl rt)
endif()

# if(target STREQUAL "Simulator_Function")
#     target_link_libraries(main funcsim)
# else()
#     target_link_libraries(main
# 			  /usr/lib/gcc-cross/aarch64-linux-gnu/libstdc++.so.6
# 			  ascendcl acl_dvpp stdc++ opencv_highgui  opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d opencv_features2d opencv_videoio)
# endif()

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
