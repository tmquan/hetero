message(${CMAKE_CURRENT_SOURCE_DIR})
# ----------------------------------------------------------------------------
find_package(CUDA REQUIRED)
# ----------------------------------------------------------------------------
if(CUDA_FOUND)
    message("CUDA is found on this system.")
	include_directories(${CUDA_INCLUDE_DIRS})
	set(CUDA_NVCC_FLAGS "
		# -gencode;arch=compute_35,code=sm_35; 
		# -gencode;arch=compute_30,code=sm_30; 
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
find_package(OpenMP REQUIRED)
# ----------------------------------------------------------------------------
if(OPENMP_FOUND)
    message("OPENMP is found on this system.")
else(OPENMP_FOUND)
    message("OPENMP is not installed on this system.")
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
# cuda_add_executable(LDAV_float_stencil_3d07 test_stencil_3d07.cu stencil_3d07.cu)
# add_executable(generator_make_stencil_3d make_stencil_3d.cpp)

# target_link_libraries(generator_stencil_3d 
	# ${MPI_LIBRARIES} 
	# ${OpenMP_CXX_FLAGS}
	# hetero_cmdparser
	# )
# ----------------------------------------------------------------------------
# add_test(Test_Generator_Make_Stencil_3D
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/generator_make_stencil_3d
	# )
	
cuda_add_executable(test_float_stencil_3d07 test_stencil_3d07.cu stencil_3d07.cu)
target_link_libraries(test_float_stencil_3d07
	${MPI_LIBRARIES} 
	${OpenMP_CXX_FLAGS}
	hetero_cmdparser
	)
add_test(LDAV_float_stencil_3d07
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 1
	${MPIEXEC_PREFLAGS} --host node001
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_float_stencil_3d07
	${MPIEXEC_POSTFLAGS} |sort)
	
