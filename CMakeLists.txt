cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 11)

project(PyEPnP)

add_subdirectory(pybind11)
#LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/pybind11/tools)
#include_directories(${PROJECT_SOURCE_DIR}/pybind11/include)
# ----------------------------------------------------------------------------
#   Find Dependencies -- opencv
# ----------------------------------------------------------------------------
find_package(OpenCV 4 REQUIRED)
IF (USE_CONTRIB)
    add_definitions(-DUSE_CONTRIB)
ENDIF ()
if (NOT OpenCV_VERSION VERSION_LESS "3.0")
    ADD_DEFINITIONS(-DOPENCV_VERSION_3)
    SET(OPENCV_VERSION_3 ON)
ELSE ()
    SET(OPENCV_VERSION_3 OFF)
ENDIF ()

include_directories(${OpenCV_INCLUDE_DIRS})

SET(REQUIRED_LIBRARIES ${REQUIRED_LIBRARIES} ${OpenCV_LIBS})

# ----------------------------------------------------------------------------
#   Find Dependencies -- python numpy
# ----------------------------------------------------------------------------
#find_package(PythonLibs 3 REQUIRED)
#include_directories(${PYTHON_INCLUDE_DIRS})
#SET(REQUIRED_LIBRARIES ${PYTHON_LIBRARIES})

include(Cmake/FindNumPy.cmake)
include_directories(${PYTHON_NUMPY_INCLUDE_DIR})

# ----------------------------------------------------------------------------
#   source
# ----------------------------------------------------------------------------
include_directories(
        ${PROJECT_SOURCE_DIR}/src/DUtils/
        ${PROJECT_SOURCE_DIR}/src/
)
SET(SOURCES
        ${PROJECT_SOURCE_DIR}/src/DUtils/Random.cpp
        ${PROJECT_SOURCE_DIR}/src/DUtils/Timestamp.cpp
        ${PROJECT_SOURCE_DIR}/src/PnPsolver.cpp
        )
SET(SOURCE_PYBIND
        ${PROJECT_SOURCE_DIR}/src/pyepnp.cpp
        ${PROJECT_SOURCE_DIR}/src/ndarray_converter.cpp
        )


pybind11_add_module(PyEPnP ${SOURCES} ${SOURCE_PYBIND})
target_link_libraries(PyEPnP PRIVATE ${REQUIRED_LIBRARIES})

