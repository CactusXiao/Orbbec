cmake_minimum_required(VERSION 3.22)

set(OrbbecSDK_ROOT ${CMAKE_CURRENT_LIST_DIR}/lib/OrbbecSDK_v2.7.2 CACHE PATH "Bundled Orbbec SDK root")
set(OrbbecSDK_DIR ${OrbbecSDK_ROOT}/lib)
set(OrbbecSDK_INCLUDE_DIRS ${OrbbecSDK_ROOT}/include)

project(orbbec)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(orbbec 
    src/sync/ego.cpp
    src/sync/fisheyes.cpp
    src/sync/tactile.cpp
    src/sync/sync.cpp
    src/sync/shared_utils.cpp
    src/sync/interactive_visualization.cpp
    src/sync/viewer.cpp
    src/sync/collection.cpp
    src/sync/calibration.cpp
    src/sync/utils/cJSON.c
    src/sync/utils/utils.cpp
    src/sync/utils/utils_c.c
    src/sync/utils/utils_opencv.cpp
)

add_executable(orbbec_probe
    src/probe_orbbec.cpp
)

find_package(OrbbecSDK REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters registration segmentation sample_consensus)
find_package(Threads REQUIRED)

target_include_directories(orbbec PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/src
    ${CMAKE_CURRENT_LIST_DIR}/src/sync
    ${CMAKE_CURRENT_LIST_DIR}/src/sync/utils
)

add_definitions(${PCL_DEFINITIONS})

target_link_libraries(orbbec PRIVATE
    ob::OrbbecSDK 
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    Threads::Threads
)

target_link_libraries(orbbec_probe PRIVATE
    ob::OrbbecSDK
)

if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_INSTALL_RPATH "@loader_path/../lib/OrbbecSDK_v2.7.2/lib")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
elseif(UNIX)
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib/OrbbecSDK_v2.7.2/lib")
    set_target_properties(orbbec PROPERTIES BUILD_RPATH "${OrbbecSDK_ROOT}/lib")
    set_target_properties(orbbec_probe PROPERTIES BUILD_RPATH "${OrbbecSDK_ROOT}/lib")
endif()

install(FILES
    ${CMAKE_CURRENT_LIST_DIR}/src/sync/config.json
    ${CMAKE_CURRENT_LIST_DIR}/src/sync/extrinsic_health_check.py
    ${CMAKE_CURRENT_LIST_DIR}/src/sync/hand_joint_gt_worker.py
    DESTINATION bin
)
install(TARGETS orbbec RUNTIME DESTINATION bin)
