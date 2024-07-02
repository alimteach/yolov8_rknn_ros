cmake_minimum_required(VERSION 3.0.2)

project(rknn_yolov8_ros)

# macro(use_cxx11)
#   if (CMAKE_VERSION VERSION_LESS "3.1")
#     if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#       set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
#     endif ()
#   else ()
#     set (CMAKE_CXX_STANDARD 17)
#   endif ()
# endmacro(use_cxx11)
# 生成配置头文件
configure_file(
  "${PROJECT_SOURCE_DIR}/include/config.h.in"
  "${PROJECT_BINARY_DIR}/include/config.h"
)

include_directories(${PROJECT_BINARY_DIR}/include)
# find_package(OpenCV REQUIRED)                              
find_package(catkin REQUIRED COMPONENTS
    roscpp
    roslib
    cv_bridge
    image_transport
    image_geometry
    sensor_msgs
    message_generation
    object_information_msgs    
)
# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -fexceptions")

# set(SOURCES src/main.cpp src/postprocess.cpp src/rk_common.cpp)

# set(EXTRA_LIBS ${OpenCV_LIBS} /usr/local/lib/librknnrt.so)

# set(EXECUTABLE_OUTPUT_PATH "../")

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES rknn_ros
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

link_directories(
  ${catkin_LIB_DIRS} 
  lib
)
include_directories("include"
  ${catkin_INCLUDE_DIRS}
  )

# add_link_options(-fopenmp -s -ldl -lpthread -pthread -lgomp -rdynamic -march=armv8.2-a)

# add_executable(yolov8_ros ${SOURCES})

# target_link_libraries(YoloV8_NPU ${EXTRA_LIBS})

add_executable(rknn_yolov8_node src/main.cpp src/postprocess.cpp src/rk_common.cpp)
add_dependencies( rknn_yolov8_node object_information_msgs_generate_messages_cpp)
target_link_libraries(rknn_yolov8_node ${catkin_LIBRARIES} librknnrt.so ${OpenCV_LIBS} ${EXTRA_LIBS})