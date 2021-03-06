# hetero/samples/mpi/hello_word
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
add_executable(mpi-io io.cpp)

target_link_libraries(mpi-io ${MPI_LIBRARIES} 
	hetero_csv
	hetero_cmdparser
	)
# ----------------------------------------------------------------------------
add_test(Test_MPI_IO_CMDPARSER
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 2
	${MPIEXEC_PREFLAGS} --host node001 
	${MPIEXEC_NUMPROC_FLAG} 2
	${MPIEXEC_PREFLAGS} --host node002 
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mpi-io 
    ${RESOURCE_INPUT_PATH}/kiwi.csv)