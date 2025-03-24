cmake_minimum_required(VERSION 3.20)

# set the project name
project(simulation)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
# set(BUILD_SHARED_LIBS OFF)
# set(CMAKE_EXE_LINKER_FLAGS "-static")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS True)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

include(FetchContent)
set(GUROBI_DIR $ENV{GUROBI_HOME})
if (NOT DEFINED ENV{GUROBI_HOME} AND NOT DEFINED GUROBI_DIR)
    message(FATAL_ERROR "No cache variable named GUROBI_DIR is defined, and GUROBI_HOME was not found in the environment variables. Please, define one of them.")
endif ()
find_package(GUROBI REQUIRED)
include_directories(${GUROBI_INCLUDE_DIRS})
message("GUROBI: " ${GUROBI_LIBRARY})
message("GUROBI: " ${GUROBI_CXX_LIBRARY})

add_subdirectory(src)
add_subdirectory(include)

enable_testing()

#add_subdirectory(test)
