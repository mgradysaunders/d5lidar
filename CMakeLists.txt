# CMake version.
cmake_minimum_required(VERSION 3.14)

# Project.
project(
    d5lidar
    VERSION 0.0.1
    DESCRIPTION "DIRSIG5 Lidar utilities"
    LANGUAGES C CXX
    )

# Set C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fno-omit-frame-pointer -fsanitize=address")

# Set release.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# If necessary, include CTest.
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif()

# Include GNUInstallDirs.
#include(GNUInstallDirs)
#message(STATUS "${PROJECT_NAME} ${PROJECT_VERSION}")
#message(STATUS "INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
#message(STATUS "INSTALL_INCLUDEDIR: ${CMAKE_INSTALL_INCLUDEDIR}")
#message(STATUS "INSTALL_LIBDIR: ${CMAKE_INSTALL_LIBDIR}")
#message(STATUS "INSTALL_BINDIR: ${CMAKE_INSTALL_BINDIR}")
#message(STATUS "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

#find_package(Eigen3 REQUIRED)
#find_package(pybind11 REQUIRED)
find_package(Python COMPONENTS Development Interpreter REQUIRED)
set(SITE_PACKAGES_DIR "python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")
set(PYTHON_MODULE_DIR "${CMAKE_INSTALL_LIBDIR}/${SITE_PACKAGES_DIR}")
message(STATUS "SITE_PACKAGES_DIR: ${SITE_PACKAGES_DIR}")
message(STATUS "PYTHON_MODULE_DIR: ${PYTHON_MODULE_DIR}")

include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)
FetchContent_Declare(
  "pybind11"
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG "v3.0.0"
  GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable("pybind11")

set(LIB_NAME d5lidar)
pybind11_add_module(
    ${LIB_NAME} 
    src/d5lidar.cpp 
    src/BinFile.cpp
    src/thirdparty/miniz.c
    )
set_target_properties(
    ${LIB_NAME}
    PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    #LIBRARY_OUTPUT_DIRECTORY "${PYTHON_MODULE_DIR}"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
    )
target_include_directories(${LIB_NAME} PUBLIC src/thirdparty)
#target_link_libraries(${LIB_NAME} PUBLIC Eigen3::Eigen)
#install(TARGETS ${LIB_NAME} DESTINATION "${PYTHON_MODULE_DIR}")
