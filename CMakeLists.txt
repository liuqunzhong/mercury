cmake_minimum_required(VERSION 2.8)
project(mercury)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_BUILD_TYPE release)

set(CMAKE_CXX_STANDARD 11)

option(USE_OPENMP "use openmp for muti-thread" ON)
#option(USE_X86 "build for x86 cpu" ON)
option(USE_CUDA "build for nvidia gpu" OFF)
option(USE_ARM "build for arm devices" OFF)
option(USE_STRING "" ON)
option(USE_STDIO "" ON)

if(USE_OPENMP)
    find_package(OpenMP)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "Found openmp in ${OPENMP_INCLUDE_DIR}")
    message(STATUS " |--openmp cflags: ${OpenMP_C_FLAGS}")
    message(STATUS " |--openmp cxxflags: ${OpenMP_CXX_FLAGS}")
    message(STATUS " |--openmp cflags: ${OpenMP_EXE_LINKER_FLAGS}")
    #ADD_DEFINITIONS(-DUSE_OPENMP)
endif()

if(USE_CUDA)
    FIND_PACKAGE(CUDA REQUIRED)
endif()

SET( CMAKE_VERBOSE_MAKEFILE ON)

add_compile_options(-std=c++11)
add_definitions(-fPIC)
add_definitions(-Ofast)
add_definitions(-ffast-math)
add_definitions(-Wall)# -Wextra  -Wno-unused-function

message(STATUS "CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR}")

#CMAKE_CURRENT_BINARY_DIR
configure_file(config.h.in ${CMAKE_SOURCE_DIR}/config.h)

include_directories(base)
include_directories(framework)
include_directories(operator)
include_directories(utils)
include_directories(./)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(/usr/local/cuda-8.0/include)

aux_source_directory(. DIRSRCS)
add_subdirectory(base)
add_subdirectory(framework)
add_subdirectory(operator)
#add_subdirectory(utils)

#set(SOURCE_FILES main.cpp)

if (USE_CUDA)
    cuda_add_executable(${PROJECT_NAME} ${DIRSRCS})
    target_link_libraries(${PROJECT_NAME} base framework operators -lm -ldl -lcuda)
else()
    add_executable(${PROJECT_NAME} ${DIRSRCS})
    target_link_libraries(${PROJECT_NAME} base framework operators -lm -ldl)
endif ()
