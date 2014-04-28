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
# find_package(OpenCV REQUIRED)
# if(OpenCV_FOUND)
    # message("OPENCV is found on this system.")
	# include_directories(${OpenCV_INCLUDE_DIR})
# else(OpenCV_FOUND)
    # message("OPENCV is not installed on this system.")
# endif()
# ----------------------------------------------------------------------------
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)
LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/lib)
# ----------------------------------------------------------------------------
cuda_add_executable(hybrid-exchange_2d exchange_2d.cu)
# add_executable(hybrid-exchange_2d exchange_2d.cpp)

target_link_libraries(hybrid-exchange_2d ${MPI_LIBRARIES} 
	hetero_cmdparser
	# ${OpenCV_LIBS}
	)
# ----------------------------------------------------------------------------
add_test(Test_Hybrid_Exchange_2D_Fittable
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 16 
	${MPIEXEC_PREFLAGS} --host node001
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-exchange_2d
	--srcFile		${RESOURCE_INPUT_PATH}/barbara_512x512.raw
	# --srcFile		${RESOURCE_INPUT_PATH}/ocean.raw
	--dimx 512 
	--dimy 512
	| sort

	)
	
add_test(Test_Hybrid_Exchange_2D_Oversubcribe
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 16 
	${MPIEXEC_PREFLAGS} --host node001
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-exchange_2d
	--srcFile		${RESOURCE_INPUT_PATH}/barbara_500x450.raw
	--dimx 500 
	--dimy 450
	| sort

	)