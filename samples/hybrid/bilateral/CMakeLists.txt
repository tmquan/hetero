# hetero/samples/hybrid/bilateral
# ----------------------------------------------------------------------------
find_package(CUDA REQUIRED)
# ----------------------------------------------------------------------------
if(CUDA_FOUND)
    message("CUDA is found on this system.")
	include_directories(${CUDA_INCLUDE_DIRS})
	set(CUDA_NVCC_FLAGS "
		-gencode;arch=compute_35,code=sm_35; 
		-gencode;arch=compute_30,code=sm_30; 
		-gencode;arch=compute_20,code=sm_20; 
		# -gencode;arch=compute_11,code=sm_11; 
		# -gencode;arch=compute_12,code=sm_12;
		# -gencode;arch=compute_13,code=sm_13;
		")

	# add -Wextra compiler flag for gcc compilations
	if (UNIX)
		# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wall;")
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--disable-warnings;")
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
	endif (UNIX)
else(CUDA_FOUND)
    message(FATAL_ERROR "CUDA is not installed on this system.")
endif()
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
cuda_add_executable(hybrid-bilateral 
					bilateral.cu)
target_link_libraries(hybrid-bilateral 
	${MPI_LIBRARIES}
	hetero_cmdparser)
# ----------------------------------------------------------------------------
add_test(Test_Hybrid_Bilateral_Strong_1_processes_Shared_Memory_System
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 1
	${MPIEXEC_PREFLAGS} --host node001
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-bilateral --dimx=512 --dimy=512 --dimz=1512
	${MPIEXEC_POSTFLAGS} |sort)
add_test(Test_Hybrid_Bilateral_Strong_2_processes_Shared_Memory_System
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 2
	${MPIEXEC_PREFLAGS} --host node001
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-bilateral --dimx=512 --dimy=512 --dimz=1512 
	${MPIEXEC_POSTFLAGS} |sort)
add_test(Test_Hybrid_Bilateral_Strong_4_processes_Shared_Memory_System
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 4
	${MPIEXEC_PREFLAGS} --host node001
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-bilateral --dimx=512 --dimy=512 --dimz=1512
	${MPIEXEC_POSTFLAGS} |sort)
add_test(Test_Hybrid_Bilateral_Strong_8_processes_Shared_Memory_System
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 4
	${MPIEXEC_PREFLAGS} --host node001
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-bilateral --dimx=512 --dimy=512 --dimz=1512 :
	${MPIEXEC_NUMPROC_FLAG} 4
	${MPIEXEC_PREFLAGS} --host node002
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-bilateral --dimx=512 --dimy=512 --dimz=1512
	${MPIEXEC_POSTFLAGS} |sort)
add_test(Test_Hybrid_Bilateral_Strong_12_processes_Shared_Memory_System
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 6
	${MPIEXEC_PREFLAGS} --host node001
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-bilateral --dimx=512 --dimy=512 --dimz=1512 :
	${MPIEXEC_NUMPROC_FLAG} 6
	${MPIEXEC_PREFLAGS} --host node002
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-bilateral --dimx=512 --dimy=512 --dimz=1512
	${MPIEXEC_POSTFLAGS} |sort)