# hetero/samples/mpi/hello_word
# ----------------------------------------------------------------------------
# All binary files will be stored here
# ----------------------------------------------------------------------------
# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
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
add_executable(mpi_hello_world mpi_hello_world.cpp)
target_link_libraries(mpi_hello_world ${MPI_LIBRARIES})
# ----------------------------------------------------------------------------

add_test(Test_MPI_Hello_World_Call_By_Host
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 10
	${MPIEXEC_PREFLAGS} --host head,node001,node002
	${EXECUTABLE_OUTPUT_PATH}/mpi_hello_world
	${MPIEXEC_POSTFLAGS} |sort)
add_test(Test_MPI_Hello_World_Call_By_Hostfile
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 28
	${MPIEXEC_PREFLAGS} --machinefile ${RESOURCE_INPUT_PATH}/garnet.unist.ac.kr
	${EXECUTABLE_OUTPUT_PATH}/mpi_hello_world
	${MPIEXEC_POSTFLAGS} | sort)
add_test(Test_MPI_Hello_World_Call_By_Worker_Master
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 8
	${MPIEXEC_PREFLAGS} --host node001 
	${EXECUTABLE_OUTPUT_PATH}/mpi_hello_world : 
	${MPIEXEC_NUMPROC_FLAG} 8
	${MPIEXEC_PREFLAGS} --host node002 
	${EXECUTABLE_OUTPUT_PATH}/mpi_hello_world : 
	${MPIEXEC_NUMPROC_FLAG} 1
	${MPIEXEC_PREFLAGS} --host head
	${EXECUTABLE_OUTPUT_PATH}/mpi_hello_world
	${MPIEXEC_POSTFLAGS} | sort)