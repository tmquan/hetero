message(${CMAKE_CURRENT_SOURCE_DIR})
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
find_package(CUDA REQUIRED)
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
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)
LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/lib)
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
add_executable(generator_make_segment_3d make_segment_3d.cpp)

# target_link_libraries(generator_stencil_3d 
	# ${MPI_LIBRARIES} 
	# ${OpenMP_CXX_FLAGS}
	# hetero_cmdparser
	# )
# ----------------------------------------------------------------------------
add_test(Test_Generator_Make_Segment_3D
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/generator_make_segment_3d
	)
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
cuda_add_executable(hybrid-app_median 
	app_median.cu
	median_3d.cu
	)

target_link_libraries(hybrid-app_median ${MPI_LIBRARIES} 
	hetero_cmdparser
	)
# ----------------------------------------------------------------------------
cuda_add_executable(hybrid-app_stddev 
	app_stddev.cu
	stddev_3d.cu
	)

target_link_libraries(hybrid-app_stddev ${MPI_LIBRARIES} 
	hetero_cmdparser
	)
# ----------------------------------------------------------------------------
cuda_add_executable(hybrid-app_bilateral  
	app_bilateral.cu
	bilateral_3d.cu
	)

target_link_libraries(hybrid-app_bilateral ${MPI_LIBRARIES} 
	hetero_cmdparser
	)
# ----------------------------------------------------------------------------
add_executable(hybrid-sys_segment sys_segment.cpp)

target_link_libraries(hybrid-sys_segment
	hetero_cmdparser
	)

add_test(Test_Hybrid_Segment_Threshold_3D
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-sys_segment 
	${RESOURCE_INPUT_PATH}/garnet.unist.ac.kr
	)
