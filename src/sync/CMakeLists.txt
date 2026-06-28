# Copyright (c) Orbbec Inc. All Rights Reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.10)
project(sync)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(OrbbecSDK_DIR ${CMAKE_CURRENT_LIST_DIR}/../../lib/OrbbecSDK_v2.7.2/lib)
find_package(OrbbecSDK REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters registration segmentation sample_consensus)
find_package(Threads REQUIRED)

add_executable(${PROJECT_NAME}
    ego.cpp
    fisheyes.cpp
    tactile.cpp
    sync.cpp
    shared_utils.cpp
    interactive_visualization.cpp
    viewer.cpp
    collection.cpp
    calibration.cpp
    utils/cJSON.c
    utils/utils.cpp
    utils/utils_c.c
    utils/utils_opencv.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/..
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/utils
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    ob::OrbbecSDK
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    Threads::Threads
)

add_definitions(${PCL_DEFINITIONS})

if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_INSTALL_RPATH "@loader_path/../lib")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
elseif(UNIX)
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib/OrbbecSDK_v2.7.2/lib")
    set_target_properties(${PROJECT_NAME} PROPERTIES BUILD_RPATH "${CMAKE_CURRENT_LIST_DIR}/../../lib/OrbbecSDK_v2.7.2/lib")
endif()

install(FILES
    ${CMAKE_CURRENT_LIST_DIR}/config.json
    ${CMAKE_CURRENT_LIST_DIR}/hand_joint_gt_worker.py
    DESTINATION bin
)
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
