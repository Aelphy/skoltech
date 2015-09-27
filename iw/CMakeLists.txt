cmake_minimum_required(VERSION 3.3)
project(face_rec)

find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp)

add_executable(face_rec ${SOURCE_FILES})
target_link_libraries(face_rec ${OpenCV_LIBS})
