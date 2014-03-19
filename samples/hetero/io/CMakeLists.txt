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
LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/lib)
add_executable(mpi-io io.cpp)
target_link_libraries(mpi-io ${MPI_LIBRARIES} ${CSV_LIB_NAME})
# ----------------------------------------------------------------------------

# add_test(Test_MPI_Hello_World_Call_By_Host
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 10
	# ${MPIEXEC_PREFLAGS} --host head,node001,node002
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mpi-io
	# ${MPIEXEC_POSTFLAGS} |sort)
# add_test(Test_MPI_Hello_World_Call_By_Hostfile
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 28
	# ${MPIEXEC_PREFLAGS} --machinefile ${RESOURCE_INPUT_PATH}/garnet.unist.ac.kr
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mpi-io
	# ${MPIEXEC_POSTFLAGS} | sort)
# add_test(Test_MPI_Hello_World_Call_By_Worker_Master
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 8
	# ${MPIEXEC_PREFLAGS} --host node001 
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mpi-io : 
	# ${MPIEXEC_NUMPROC_FLAG} 8
	# ${MPIEXEC_PREFLAGS} --host node002 
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mpi-io : 
	# ${MPIEXEC_NUMPROC_FLAG} 1
	# ${MPIEXEC_PREFLAGS} --host head
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mpi-io
	# ${MPIEXEC_POSTFLAGS} | sort)