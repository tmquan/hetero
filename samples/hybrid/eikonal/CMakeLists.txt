# hetero/samples/hybrid/eikonal
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
cuda_add_executable(hybrid-golden_eikonal hybrid-golden_eikonal.cu)
target_link_libraries(hybrid-golden_eikonal ${MPI_LIBRARIES})
# ----------------------------------------------------------------------------
add_test(Test_Hybrid_Eikonal_Golden
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 1
	${MPIEXEC_PREFLAGS} --host node001 # node001 has GPUs
	# ${MPIEXEC_PREFLAGS} -nolocal  
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-golden_eikonal
	${MPIEXEC_POSTFLAGS} |sort)
