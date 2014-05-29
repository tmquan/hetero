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
##############################################################################
# <summary> Case study for Fourier Transform.	</summary>
set(FOURIER			
	dft.cu
	test_dft.cu)

# Link to CUDA
CUDA_ADD_EXECUTABLE(test_dft ${FOURIER})
CUDA_ADD_CUFFT_TO_TARGET(test_dft)
CUDA_ADD_CUBLAS_TO_TARGET(test_dft)
target_link_libraries(test_dft ${MPI_LIBRARIES} 
	hetero_cmdparser
	)
add_test(Test_CMIG14_Fourier
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 1
	${MPIEXEC_PREFLAGS} --host node001
	# ${MPIEXEC_PREFLAGS} --machinefile 
	# ${RESOURCE_INPUT_PATH}/garnet.unist.ac.kr
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_dft
	--dimx=128 
	--dimy=128 
	--dimz=256 
	--srcFDFT=${RESOURCE_INPUT_PATH}/lenna_128x128x256_full.bin
	--dstFDFT=${RESOURCE_INPUT_PATH}/lenna_128x128x256_fdft.bin
	--dstIDFT=${RESOURCE_INPUT_PATH}/lenna_128x128x256_idft.bin
	)
##############################################################################
# <summary> Case study for Wavelet Transform.	</summary>
set(WAVELET			
	dwt.cu
	test_dwt.cu)

# Link to CUDA
CUDA_ADD_EXECUTABLE(test_dwt ${WAVELET})
CUDA_ADD_CUFFT_TO_TARGET(test_dwt)
CUDA_ADD_CUBLAS_TO_TARGET(test_dwt)

target_link_libraries(test_dwt ${MPI_LIBRARIES} 
	hetero_cmdparser
	)
add_test(Test_CMIG14_Wavelet
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 1
	${MPIEXEC_PREFLAGS} --host node001
	# ${MPIEXEC_PREFLAGS} --machinefile 
	# ${RESOURCE_INPUT_PATH}/garnet.unist.ac.kr
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_dwt
	--dimx=128 
	--dimy=128 
	--dimz=256 
	--srcFDWT=${RESOURCE_INPUT_PATH}/lenna_128x128x256_full.bin
	--dstFDWT=${RESOURCE_INPUT_PATH}/lenna_128x128x256_fdwt.bin
	--dstIDWT=${RESOURCE_INPUT_PATH}/lenna_128x128x256_idwt.bin
	)
##############################################################################
set(OPERATORS
	add.cu
	sub.cu
	mul.cu
	div.cu
	dxt.cu
	dyt.cu
	dzt.cu
	dft.cu
	dwt.cu
	shrink.cu
	)
##############################################################################
# <summary> Case study for CSMRI25D with 1 gpu.	</summary>
# <summary> Using Richardson Iteration.	</summary>

# Link to CUDA
CUDA_ADD_EXECUTABLE(test_csmri25d_1gpu_richardson 
	${OPERATORS}
	test_csmri25d_1gpu_richardson.cu
	)
CUDA_ADD_CUFFT_TO_TARGET(test_csmri25d_1gpu_richardson)
CUDA_ADD_CUBLAS_TO_TARGET(test_csmri25d_1gpu_richardson)

# Link to MPI
target_link_libraries(test_csmri25d_1gpu_richardson
	${MPI_LIBRARIES} 
	${OpenMP_CXX_FLAGS}
	hetero_cmdparser
	)

# Add a test:
add_test(Test_CMIG14_CSMRI25D_1GPU_RICHARDSON 
${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 1 
	${MPIEXEC_PREFLAGS} --host node001 
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_csmri25d_1gpu_richardson 
		--spec=${RESOURCE_INPUT_PATH}/tumor1_128x128x256_kspace3.bin 			
		--mask=${RESOURCE_INPUT_PATH}/mask_x4_128x128x256.bin		
		--full=${RESOURCE_INPUT_PATH}/tumor_128x128x256_full.bin 			
		--zero=${RESOURCE_INPUT_PATH}/tumor_128x128x256_zeros.bin 			
		--dest=${RESOURCE_INPUT_PATH}/tumor_128x128x256_cs_x4.bin			
		--dimx=128			
		--dimy=128			
		--dimz=256			
		--devs=1			
		--Mu=0.1				
		--Lambda_w=0.001		
		--Lambda_t=1000		
		--Gamma_w=0.001		
		--Gamma_t=1000 
		--Omega_w=0.001		
		--Omega_t=1000 
		--Epilon=0.7			
		--nOuter=8		
		--nInner=32			
		--nLoops=8)
##############################################################################
# <summary> Case study for CSMRI25D with 1 gpu.	</summary>
# <summary> Using Conjugate Gradient.	</summary>
# Link to CUDA
CUDA_ADD_EXECUTABLE(test_csmri25d_1gpu_conjgrad 
	${OPERATORS}
	test_csmri25d_1gpu_conjgrad.cu)
CUDA_ADD_CUFFT_TO_TARGET(test_csmri25d_1gpu_conjgrad)
CUDA_ADD_CUBLAS_TO_TARGET(test_csmri25d_1gpu_conjgrad)

# Link to MPI
target_link_libraries(test_csmri25d_1gpu_conjgrad
	${MPI_LIBRARIES} 
	${OpenMP_CXX_FLAGS}
	hetero_cmdparser
	)

# Add a test:
add_test(Test_CMIG14_CSMRI25D_1GPU_CONJGRAD
${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 1 
	${MPIEXEC_PREFLAGS} --host node001 
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_csmri25d_1gpu_conjgrad 
		--spec=${RESOURCE_INPUT_PATH}/tumor1_128x128x256_kspace3.bin 			
		--mask=${RESOURCE_INPUT_PATH}/mask_x4_128x128x256.bin		
		--full=${RESOURCE_INPUT_PATH}/tumor_128x128x256_full.bin 			
		--zero=${RESOURCE_INPUT_PATH}/tumor_128x128x256_zeros.bin 			
		--dest=${RESOURCE_INPUT_PATH}/tumor_128x128x256_cs_x4.bin			
		--dimx=128			
		--dimy=128			
		--dimz=256			
		--devs=1			
		--Mu=0.1				
		--Lambda_w=0.001		
		--Lambda_t=1000		
		--Gamma_w=0.001		
		--Gamma_t=1000 
		--Omega_w=0.001		
		--Omega_t=1000 
		--Epilon=0.7			
		--nOuter=8		
		--nInner=32			
		--nLoops=8)
##############################################################################
# <summary> Case study for CSMRI25D with 1 gpu.	</summary>
# <summary> Using Single Iteration.	</summary>
# Link to CUDA
CUDA_ADD_EXECUTABLE(test_csmri25d_1gpu_singleiter 
	${OPERATORS}
	test_csmri25d_1gpu_singleiter.cu)
CUDA_ADD_CUFFT_TO_TARGET(test_csmri25d_1gpu_singleiter)
CUDA_ADD_CUBLAS_TO_TARGET(test_csmri25d_1gpu_singleiter)

# Link to MPI
target_link_libraries(test_csmri25d_1gpu_singleiter
	${MPI_LIBRARIES} 
	${OpenMP_CXX_FLAGS}
	hetero_cmdparser
	)

# Add a test:
add_test(Test_CMIG14_CSMRI25D_1GPU_SINGLEITER
${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 1 
	${MPIEXEC_PREFLAGS} --host node001 
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_csmri25d_1gpu_singleiter 
		--spec=${RESOURCE_INPUT_PATH}/tumor1_128x128x256_kspace3.bin 			
		--mask=${RESOURCE_INPUT_PATH}/mask_x4_128x128x256.bin		
		--full=${RESOURCE_INPUT_PATH}/tumor_128x128x256_full.bin 			
		--zero=${RESOURCE_INPUT_PATH}/tumor_128x128x256_zeros.bin 			
		--dest=${RESOURCE_INPUT_PATH}/tumor_128x128x256_cs_x4.bin			
		--dimx=128			
		--dimy=128			
		--dimz=256			
		--devs=1			
		--Mu=0.1				
		--Lambda_w=0.1		
		--Lambda_t=10		
		--Gamma_w=0.0001		
		--Gamma_t=10000 
		--Omega_w=0.0001		
		--Omega_t=10000 
		--Epilon=0.7			
		--nOuter=256		
		--nInner=8			
		--nLoops=8)
##############################################################################