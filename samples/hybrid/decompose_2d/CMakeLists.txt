message(${CMAKE_CURRENT_SOURCE_DIR})
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
cuda_add_executable(mpi_decompose_2d decompose_2d.cu)

target_link_libraries(mpi_decompose_2d ${MPI_LIBRARIES} 
	hetero_cmdparser
	)
# ----------------------------------------------------------------------------
add_test(Test_Hybrid_Decompose_2D_Fittable
	${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 4
	# ${MPIEXEC_PREFLAGS} --host head 
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mpi_decompose_2d
	# --srcFile		${RESOURCE_INPUT_PATH}/barbara_512x512.raw
	# --dimx 500 
	# --dimy 450
	:
	${MPIEXEC_NUMPROC_FLAG} 2 
	${MPIEXEC_PREFLAGS} --host node001  
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mpi_decompose_2d
	--srcFile		${RESOURCE_INPUT_PATH}/barbara_512x512.raw
	--dimx 512 
	--dimy 512
	:
	${MPIEXEC_NUMPROC_FLAG} 2 
	${MPIEXEC_PREFLAGS} --host node002 
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mpi_decompose_2d 
	--srcFile		${RESOURCE_INPUT_PATH}/barbara_512x512.raw 
	--dimx 512 
	--dimy 512
	| sort
	)