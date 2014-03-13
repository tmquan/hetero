# ----------------------------------------------------------------------------
# Top level Hetero project
# Author		:	Tran Minh Quan
# Email			: 	quantm@unist.ac.kr,	quantm@kaist.ac.kr
# Address		: 	High-performance Visual Computing Lab
# Website		:	http://hvcl.unist.ac.kr/
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
# ----------------------------------------------------------------------------
project(Hetero CXX C)

# ----------------------------------------------------------------------------
# All binary files will be stored here
# ----------------------------------------------------------------------------
set(RESOURCE_INPUT_PATH    "${CMAKE_SOURCE_DIR}/data")

# set(EXECUTABLE_OUTPUT_PATH 			"${CMAKE_SOURCE_DIR}/bin")

# set(CMAKE_RESOURCE_INPUT_DIRECTORY 	"${CMAKE_SOURCE_DIR}/data")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY 	"${CMAKE_SOURCE_DIR}/bin")
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY 	"${CMAKE_SOURCE_DIR}/lib")

# set(CMAKE_RESOURCE_INPUT_DIRECTORY 	"${CMAKE_SOURCE_DIR}/data")
# set(RUNTIME_OUTPUT_DIRECTORY 			"${CMAKE_SOURCE_DIR}/bin")
# set(LIBRARY_OUTPUT_DIRECTORY 			"${CMAKE_SOURCE_DIR}/lib")


# SET(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)
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

# Various data that is used by hetero libraries and/or demo applications.
add_subdirectory(data)

# Examples
add_subdirectory(samples)
 
# ----------------------------------------------------------------------------
# Clean the output and test logs
# ----------------------------------------------------------------------------
add_custom_target(clear 
	COMMAND rm			
	`find ./ -iname '*Make*' -not -name CMakeLists.txt` 
	`find ./ -iname 'bin'`
	`find ./ -iname 'Testing'`
	-rf)    
# add_custom_target(clear 
	# COMMAND COMMAND ${CMAKE_COMMAND} -P clear.cmake)    
# set(cmake_generated ${CMAKE_SOURCE_DIR}/CMakeCache.txt
                    # ${CMAKE_SOURCE_DIR}/cmake_install.cmake  
                    # ${CMAKE_SOURCE_DIR}/Makefile
                    # ${CMAKE_SOURCE_DIR}/CMakeFiles
                    # # ${CMAKE_SOURCE_DIR}/bin
# )

# foreach(file ${cmake_generated})
	# if (EXISTS ${file})
		# file(REMOVE_RECURSE ${file})
	# endif()
# endforeach(file)