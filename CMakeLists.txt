cmake_minimum_required(VERSION 3.22)

set(OrbbecSDK_DIR /home/ubuntu/orbbec/lib/OrbbecSDK_v2.7.2/lib)
set(OrbbecSDK_INCLUDE_DIRS /home/ubuntu/orbbec/lib/OrbbecSDK_v2.7.2/include)

project(orbbec)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(orbbec 
    src/sync/sync.cpp
    src/sync/fisheyes.cpp
    src/sync/tactile.cpp
    src/sync/PipelineHolder.cpp
    src/sync/FramePairingManager.cpp
    src/sync/utils/utils.cpp
    src/sync/utils/cJSON.c
    src/sync/utils/utils_c.c
    src/sync/utils/utils_opencv.cpp
)

find_package(OrbbecSDK REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters registration)
find_package(Threads REQUIRED)
#find_package(cJSON REQUIRED)
#target_include_directories(orbbec PRIVATE /usr/local/include)
#target_include_directories(orbbec PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(orbbec PRIVATE
    ${OrbbecSDK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${cJSON_INCLUDE_DIRS}
    src
    src/sync/utils
    src/sync/
)

add_definitions(${PCL_DEFINITIONS})

target_link_libraries(orbbec 
    ob::OrbbecSDK 
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    Threads::Threads
)
