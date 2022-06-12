cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(PCL_FRAGMENTS)

find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS} include)
link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})
add_executable(visualize just_visualize.cpp)
target_link_libraries(visualize ${PCL_LIBRARIES})
add_executable(regions region_grow.cpp)
target_link_libraries(regions ${PCL_LIBRARIES})
# set_property(TARGET visualize PROPERTY CXX_STANDARD 11)