# hetero/samples/mpi/io
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
add_executable(mpi-io io.cpp)
target_link_libraries(mpi-io ${MPI_LIBRARIES})
# ----------------------------------------------------------------------------
add_test(Test_MPI_IO_16_PROCESSES
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 16
	${MPIEXEC_PREFLAGS} --host node001,node002
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mpi-io
	${MPIEXEC_POSTFLAGS} |sort)