# Acknowledgment:
# This subroutine is a C/C++ version which reimplemented the series of neural network
# including back propagation network by Ryan Harris
# More details can be found in the author channel.
# http://www.youtube.com/user/nqramjets/
# ----------------------------------------------------------------------------
find_package(OpenMP REQUIRED)
# ----------------------------------------------------------------------------
if(OPENMP_FOUND)
	message("OpenMP is found on this system.")
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else(OPENMP_FOUND)
    message("OpenMP is not installed on this system.")
endif()
# ----------------------------------------------------------------------------
find_package(CUDA REQUIRED)
# ----------------------------------------------------------------------------
if(CUDA_FOUND)
    message("CUDA is found on this system.")
	include_directories(${CUDA_INCLUDE_DIRS})
	set(CUDA_NVCC_FLAGS "
		-Xcompiler ${OpenMP_C_FLAGS}
		-Xcompiler ${OpenMP_CXX_FLAGS}
		# -gencode;arch=compute_35,code=sm_35; 
		# -gencode;arch=compute_30,code=sm_30; 
		# -gencode;arch=compute_21,code=sm_21; 
		-gencode;arch=compute_20,code=sm_20; 
		# -gencode;arch=compute_11,code=sm_11; 
		# -gencode;arch=compute_12,code=sm_12;
		# -gencode;arch=compute_13,code=sm_13;
		# -Xptxas -dlcm=cg;
		# -Xptxas -dlcm=ca;
		# -lineinfo;
		# -O3;
		# Caching 
		# Default mode 
			# Attempts to hit in L1, then L2, then GMEM 
			# Load granularity is 128-byte line 
		# Non-caching 
			# Compile with –Xptxas –dlcm=cg option to nvcc
			# Attempts to hit in L2, then GMEM 
			# Do not hit in L1, invalidate the line if it’s in L1 already 
			# Load granularity is 32-bytes
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
# <summary> Case study for Fourier Transform.	</summary>
set(NEURAL_NETWORKS			
	backpropagation.cu
	app_neural_network.cu)

# Link to CUDA
CUDA_ADD_EXECUTABLE(test_neural_network ${NEURAL_NETWORKS})
CUDA_ADD_CUFFT_TO_TARGET(test_neural_network)
CUDA_ADD_CUBLAS_TO_TARGET(test_neural_network)
target_link_libraries(test_neural_network 
	${MPI_LIBRARIES} 
	hetero_cmdparser
	)
add_test(Test_Neural_Network
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 1
	${MPIEXEC_PREFLAGS} --host node001
	# ${MPIEXEC_PREFLAGS} --machinefile 
	# ${RESOURCE_INPUT_PATH}/garnet.unist.ac.kr
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_neural_network
	)