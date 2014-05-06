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
find_package(HDF5)
if(HDF5_FOUND)
	message("HDF5 is found on this system.")
	include_directories(${HDF5_INCLUDE_DIRS})
else(HDF5_FOUND)
    message("HDF5 is not installed on this system.")
endif()
# ----------------------------------------------------------------------------
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)
LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/lib)
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# add_executable(hybrid-app_hello_world_3d app_hello_world_3d.cpp)

# target_link_libraries(hybrid-app_hello_world_3d ${MPI_LIBRARIES} 
	# hetero_cmdparser
	# )	
# # ----------------------------------------------------------------------------
# add_executable(hybrid-sys_hello_world_3d sys_hello_world_3d.cpp)

# target_link_libraries(hybrid-sys_hello_world_3d
	# hetero_cmdparser
	# )

# add_test(Test_Hybrid_HelloWorld_System_3d
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-sys_hello_world_3d 
	# ${RESOURCE_INPUT_PATH}/garnet.unist.ac.kr
	# )
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
add_executable(hybrid-app_io_strip_3d app_io_strip_3d.cpp)

target_link_libraries(hybrid-app_io_strip_3d ${MPI_LIBRARIES} 
	hetero_cmdparser
	)
# ----------------------------------------------------------------------------
add_executable(hybrid-sys_io_strip_3d sys_io_strip_3d.cpp)

target_link_libraries(hybrid-sys_io_strip_3d
	hetero_cmdparser
	)

add_test(Test_Hybrid_IO_Strip_3D
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-sys_io_strip_3d 
	${RESOURCE_INPUT_PATH}/garnet.unist.ac.kr
	)
