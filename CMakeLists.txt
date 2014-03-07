# --------------------------------------------------------------
# Top level Hetero project
# Author		:	Tran Minh Quan
# Email			: 	quantm@unist.ac.kr,	quantm@kaist.ac.kr
# Address		: 	High-performance Visual Computing Lab
# Website		:	http://hvcl.unist.ac.kr/quantm/
# --------------------------------------------------------------
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
# --------------------------------------------------------------
project(Hetero CXX C)

# ----------------------------------------------------------------------------
# All binary files will be stored here
# ----------------------------------------------------------------------------
set(RESOURCE_INPUT_PATH    "${PROJECT_BINARY_DIR}/data")
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin")

# set(CMAKE_RESOURCE_INPUT_DIRECTORY "${PROJECT_BINARY_DIR}/data")
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

# SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
# set(RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ----------------------------------------------------------------------------
# Enable nightly test
# ----------------------------------------------------------------------------
enable_testing()

# ----------------------------------------------------------------------------
# Process subdirectories
# ----------------------------------------------------------------------------
# hetero.hpp and legacy headers
add_subdirectory(include)

# Hetero modules
add_subdirectory(modules)

# Generate targets for documentation
add_subdirectory(doc)

# various data that is used by hetero libraries and/or demo applications.
add_subdirectory(data)

# examples
add_subdirectory(samples)

# set_target_properties(
    # PROPERTIES
    # ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    # LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    # RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
# )
# ----------------------------------------------------------------------------
# Clean the output and test logs
# ----------------------------------------------------------------------------
add_custom_target(clear 
	COMMAND rm			
	`find -iname '*Make*' -not -name CMakeLists.txt` 
	`find -iname 'bin'`
	`find -iname 'Testing'`
	-rf)    