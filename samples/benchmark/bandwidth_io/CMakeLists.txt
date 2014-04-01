# hetero/samples/benchmark/bandwidth_io
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
add_executable(hetero_bandwidth_io bandwidth_io.cpp)

target_link_libraries(hetero_bandwidth_io ${MPI_LIBRARIES} 
	hetero_cmdparser
	)
# ----------------------------------------------------------------------------
add_test(Test_Benchmark_Bandwidth_IO
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 4
	${MPIEXEC_PREFLAGS} --host head 
	# ${MPIEXEC_NUMPROC_FLAG} 2
	# ${MPIEXEC_PREFLAGS} --host node001 
	# ${MPIEXEC_NUMPROC_FLAG} 2
	# ${MPIEXEC_PREFLAGS} --host node002 
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hetero_bandwidth_io 10 10 2 2 
	)