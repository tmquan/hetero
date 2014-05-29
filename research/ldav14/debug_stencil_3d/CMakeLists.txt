message(${CMAKE_CURRENT_SOURCE_DIR})
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
		-Xptxas -dlcm=ca;
		-lineinfo;
		-O3;
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
# cuda_add_executable(LDAV_float_stencil_3d test_stencil_3d.cu stencil_3d.cu)
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
	
cuda_add_executable(debug_stencil_3d 
	test_stencil_3d.cu 
	# stencil_3d.cu
	# stencil_3d_naive_7points.cu
	)
target_link_libraries(debug_stencil_3d
	${MPI_LIBRARIES} 
	${OpenMP_CXX_FLAGS}
	hetero_cmdparser
	)
add_test(Debug_Stencil_3d
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 1
	${MPIEXEC_PREFLAGS} --host node002
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/debug_stencil_3d --ilp=8 
	${MPIEXEC_POSTFLAGS} |sort)
# ----------------------------------------------------------------------------
cuda_add_executable(debug_real_stencil_3d 
	test_real_stencil_3d.cu 
	# stencil_3d_naive_7points.cu
	)
target_link_libraries(debug_real_stencil_3d
	${MPI_LIBRARIES} 
	${OpenMP_CXX_FLAGS}
	hetero_cmdparser
	)
add_test(Debug_Real_Stencil_3d
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 1
	${MPIEXEC_PREFLAGS} --host node001
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/debug_real_stencil_3d
	${MPIEXEC_POSTFLAGS} |sort)
# ----------------------------------------------------------------------------
cuda_add_executable(debug_professor_stencil_3d 
	stencil_3d_professor.cu
	)
# target_link_libraries(debug_real_stencil_3d
	# ${MPI_LIBRARIES} 
	# ${OpenMP_CXX_FLAGS}
	# hetero_cmdparser
	# )
add_test(Debug_Professor_Stencil_3d
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 1
	${MPIEXEC_PREFLAGS} --host node001
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/debug_professor_stencil_3d 
	${MPIEXEC_POSTFLAGS} |sort)
	