message(${CMAKE_CURRENT_SOURCE_DIR})
# ----------------------------------------------------------------------------
find_package(MPI REQUIRED)
# ----------------------------------------------------------------------------
if(MPI_FOUND)
	message("MPI is found on this system.")
	include_directories(${MPI_INCLUDE_PATH})
else(MPI_FOUND)
    message(FATAL_ERROR "MPI is not installed on this system.")
endif()
# ----------------------------------------------------------------------------
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)
LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/lib)
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
add_executable(hybrid-app_hello_world_2d app_hello_world_2d.cpp)

target_link_libraries(hybrid-app_hello_world_2d ${MPI_LIBRARIES} 
	hetero_cmdparser
	)	
# ----------------------------------------------------------------------------
add_executable(hybrid-sys_hello_world_2d sys_hello_world_2d.cpp)

target_link_libraries(hybrid-sys_hello_world_2d
	hetero_cmdparser
	)

add_test(Test_Hybrid_HelloWorld_System_2d
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-sys_hello_world_2d 
	${RESOURCE_INPUT_PATH}/garnet.unist.ac.kr
	)
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
add_executable(hybrid-app_io_2d app_io_2d.cpp)

target_link_libraries(hybrid-app_io_2d ${MPI_LIBRARIES} 
	hetero_cmdparser
	)
# ----------------------------------------------------------------------------
add_executable(hybrid-sys_io_2d sys_io_2d.cpp)

target_link_libraries(hybrid-sys_io_2d
	hetero_cmdparser
	)

add_test(Test_Hybrid_IO_System_2D
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-sys_io_2d 
	${RESOURCE_INPUT_PATH}/garnet.unist.ac.kr
	)
