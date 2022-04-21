project(radar_processor)
cmake_minimum_required(VERSION 3.4)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -W -Wno-unused-parameter -O2")

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(OMP_NUM_THREADS 8)

find_package(OpenCV 3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS
    system
    filesystem
)
find_package(pybind11 CONFIG)

include_directories(
	include
	${OpenCV_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIR}
    ${pybind11_INCLUDE_DIR}
)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()


pybind11_add_module(radar_processor src/radar_processor.cpp
                                    src/radar_utils.cpp
                                    src/features.cpp)


target_link_libraries(${PROJECT_NAME} PRIVATE
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
    Eigen3::Eigen
)
