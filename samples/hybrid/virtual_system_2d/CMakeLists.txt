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
add_executable(hybrid-hello_world_virtual_2d hello_world_virtual_2d.cpp)

target_link_libraries(hybrid-hello_world_virtual_2d ${MPI_LIBRARIES} 
	hetero_cmdparser
	)

add_test(Test_Hybrid_Virtual_System_2d
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 8 
	${MPIEXEC_PREFLAGS} --host node001
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-hello_world_virtual_2d
	${RESOURCE_INPUT_PATH}/garnet.unist.ac.kr
	--runId=1 
	--virtualDimx=4 
	--virtualDimy=4 
	| sort
	)
	
# ----------------------------------------------------------------------------
add_executable(hybrid-system_2d system_2d.cpp)

target_link_libraries(hybrid-system_2d
	hetero_cmdparser
	)

add_test(Test_Hybrid_Wrapper_System_2d
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-system_2d 
	${RESOURCE_INPUT_PATH}/garnet.unist.ac.kr
	)