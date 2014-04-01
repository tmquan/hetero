	# hetero/samples/hybrid/heatflow
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
cuda_add_executable(hybrid-heatflow 
					cmdparser.cpp
					heatflow.cu)
target_link_libraries(hybrid-heatflow ${MPI_LIBRARIES})

# ----------------------------------------------------------------------------
add_test(Test_Hybrid_HeatFlow_Strong_1_processes_Distributed_Memory_System
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 1
	${MPIEXEC_PREFLAGS} --host node001
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=1512 
	${MPIEXEC_POSTFLAGS} |sort)
add_test(Test_Hybrid_HeatFlow_Strong_2_processes_Distributed_Memory_System
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 1
	${MPIEXEC_PREFLAGS} --host node001
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=1512 :
	${MPIEXEC_NUMPROC_FLAG} 1
	${MPIEXEC_PREFLAGS} --host node002
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=1512 
	${MPIEXEC_POSTFLAGS} |sort)
add_test(Test_Hybrid_HeatFlow_Strong_4_processes_Distributed_Memory_System
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 2
	${MPIEXEC_PREFLAGS} --host node001
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=1512 :
	${MPIEXEC_NUMPROC_FLAG} 2
	${MPIEXEC_PREFLAGS} --host node002
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=1512 
	${MPIEXEC_POSTFLAGS} |sort)
add_test(Test_Hybrid_HeatFlow_Strong_8_processes_Distributed_Memory_System
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 4
	${MPIEXEC_PREFLAGS} --host node001
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=1512 :
	${MPIEXEC_NUMPROC_FLAG} 4
	${MPIEXEC_PREFLAGS} --host node002
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=1512 
	${MPIEXEC_POSTFLAGS} |sort)
add_test(Test_Hybrid_HeatFlow_Strong_12_processes_Distributed_Memory_System
	${MPIEXEC} 
	${MPIEXEC_NUMPROC_FLAG} 6
	${MPIEXEC_PREFLAGS} --host node001
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=1512 :
	${MPIEXEC_NUMPROC_FLAG} 6
	${MPIEXEC_PREFLAGS} --host node002
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=1512 
	${MPIEXEC_POSTFLAGS} |sort)
	
# ----------------------------------------------------------------------------	
# add_test(Test_Hybrid_HeatFlow_Strong_1_processes_Distributed_Memory_System
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 1
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=1512 
	# ${MPIEXEC_POSTFLAGS} |sort)
# add_test(Test_Hybrid_HeatFlow_Strong_2_processes_Distributed_Memory_System
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 1
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=1512  :
	# ${MPIEXEC_NUMPROC_FLAG} 1
	# ${MPIEXEC_PREFLAGS} --host node002
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=1512  
	# ${MPIEXEC_POSTFLAGS} |sort)
# add_test(Test_Hybrid_HeatFlow_Strong_4_processes_Distributed_Memory_System
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 2
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=1512  :
	# ${MPIEXEC_NUMPROC_FLAG} 2
	# ${MPIEXEC_PREFLAGS} --host node002
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=1512  
	# ${MPIEXEC_POSTFLAGS} |sort)
# add_test(Test_Hybrid_HeatFlow_Strong_8_processes_Distributed_Memory_System
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 4
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=1512  :
	# ${MPIEXEC_NUMPROC_FLAG} 4
	# ${MPIEXEC_PREFLAGS} --host node002
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=1512  
	# ${MPIEXEC_POSTFLAGS} |sort)
# add_test(Test_Hybrid_HeatFlow_Strong_12_processes_Distributed_Memory_System
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 6
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=1512  :
	# ${MPIEXEC_NUMPROC_FLAG} 6
	# ${MPIEXEC_PREFLAGS} --host node002
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=1512  
	# ${MPIEXEC_POSTFLAGS} |sort)
	
# cuda_add_executable(hybrid-heatflow_real 
					# cmdparser.cpp
					# hybrid-heatflow_real.cu)
# target_link_libraries(hybrid-heatflow_real ${MPI_LIBRARIES})
# ----------------------------------------------------------------------------
# # Test the real, not synthetic kernel
# add_test(Test_Hybrid_HeatFlow_Real_Strong_1_processes
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 1
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow_real --dimx=512 --dimy=512 --dimz=512 
	# ${MPIEXEC_POSTFLAGS} |sort)
# add_test(Test_Hybrid_HeatFlow_Real_Strong_2_processes
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 2
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow_real --dimx=512 --dimy=512 --dimz=512 
	# ${MPIEXEC_POSTFLAGS} |sort)
# add_test(Test_Hybrid_HeatFlow_Real_Strong_4_processes
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 4
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow_real --dimx=512 --dimy=512 --dimz=512 
	# ${MPIEXEC_POSTFLAGS} |sort)
# add_test(Test_Hybrid_HeatFlow_Real_Strong_8_processes
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 4
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow_real --dimx=512 --dimy=512 --dimz=512 :
    # ${MPIEXEC_NUMPROC_FLAG} 4
	# ${MPIEXEC_PREFLAGS} --host node002
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow_real --dimx=512 --dimy=512 --dimz=512 
	# ${MPIEXEC_POSTFLAGS} |sort)

    
# add_test(Test_Hybrid_HeatFlow_Real_Weak_1_processes
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 1
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow_real --dimx=512 --dimy=512 --dimz=32 
	# ${MPIEXEC_POSTFLAGS} |sort)
# add_test(Test_Hybrid_HeatFlow_Real_Weak_2_processes
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 2
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow_real --dimx=512 --dimy=512 --dimz=64 
	# ${MPIEXEC_POSTFLAGS} |sort)
# add_test(Test_Hybrid_HeatFlow_Real_Weak_4_processes
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 4
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow_real --dimx=512 --dimy=512 --dimz=128 
	# ${MPIEXEC_POSTFLAGS} |sort)
# add_test(Test_Hybrid_HeatFlow_Real_Weak_8_processes
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 4
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow_real --dimx=512 --dimy=512 --dimz=256 :
    # ${MPIEXEC_NUMPROC_FLAG} 4
	# ${MPIEXEC_PREFLAGS} --host node002
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow_real --dimx=512 --dimy=512 --dimz=256 :
	# ${MPIEXEC_POSTFLAGS} |sort)
# ----------------------------------------------------------------------------
# add_test(Test_Hybrid_HeatFlow_Strong_1_processes
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 1
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=1512 
	# ${MPIEXEC_POSTFLAGS} |sort)
# add_test(Test_Hybrid_HeatFlow_Strong_2_processes
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 2
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=1512  
	# ${MPIEXEC_POSTFLAGS} |sort)
# add_test(Test_Hybrid_HeatFlow_Strong_4_processes
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 4
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=1512 
	# ${MPIEXEC_POSTFLAGS} |sort)
# add_test(Test_Hybrid_HeatFlow_Strong_8_processes
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 8
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=1512 
	# ${MPIEXEC_POSTFLAGS} |sort)
 # add_test(Test_Hybrid_HeatFlow_Strong_8_processes
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 4   
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=896  :
	# ${MPIEXEC_NUMPROC_FLAG} 4
	# ${MPIEXEC_PREFLAGS} --host node002
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=896 
	# ${MPIEXEC_POSTFLAGS} |sort)
 # add_test(Test_Hybrid_HeatFlow_Strong_14_processes
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 7   
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=896  :
	# ${MPIEXEC_NUMPROC_FLAG} 7
	# ${MPIEXEC_PREFLAGS} --host node002
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=896 
	# ${MPIEXEC_POSTFLAGS} |sort)

# ----------------------------------------------------------------------------
# add_test(Test_Hybrid_HeatFlow_Weak_1_processes
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 1
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=256 
	# ${MPIEXEC_POSTFLAGS} |sort)
# add_test(Test_Hybrid_HeatFlow_Weak_2_processes
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 2
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=512
	# ${MPIEXEC_POSTFLAGS} |sort)
# add_test(Test_Hybrid_HeatFlow_Weak_4_processes
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 4
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=1024
	# ${MPIEXEC_POSTFLAGS} |sort)
# add_test(Test_Hybrid_HeatFlow_Weak_8_processes
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 8
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=2048
	# ${MPIEXEC_POSTFLAGS} |sort)
	
# add_test(Test_Hybrid_HeatFlow_Weak_8_processes
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 4   
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=4096   :
	# ${MPIEXEC_NUMPROC_FLAG} 4
	# ${MPIEXEC_PREFLAGS} --host node002
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=4096 
	# ${MPIEXEC_POSTFLAGS} |sort)
# add_test(Test_Hybrid_HeatFlow_Weak_14_processes
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 7   
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=7168   :
	# ${MPIEXEC_NUMPROC_FLAG} 7
	# ${MPIEXEC_PREFLAGS} --host node002
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=7168 
	# ${MPIEXEC_POSTFLAGS} |sort)
						
			
# ----------------------------------------------------------------------------
# add_test(Test_Hybrid_HeatFlow
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 4 
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=512 :
	# ${MPIEXEC_NUMPROC_FLAG} 4
	# ${MPIEXEC_PREFLAGS} --host node002
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=512
	# ${MPIEXEC_POSTFLAGS} |sort)
# add_test(Test_Hybrid_HeatFlow_1_process
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 1
	# ${MPIEXEC_PREFLAGS} --host node002
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=512 
	# ${MPIEXEC_POSTFLAGS} |sort)
# add_test(Test_Hybrid_HeatFlow_2_processes
	# ${MPIEXEC} 
	# # ${MPIEXEC_NUMPROC_FLAG} 1   
	# # ${MPIEXEC_PREFLAGS} --host node001
	# # ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=512  :
	# ${MPIEXEC_NUMPROC_FLAG} 2
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=512 
	# ${MPIEXEC_POSTFLAGS} |sort)
# add_test(Test_Hybrid_HeatFlow_4_processes
	# ${MPIEXEC} 
	# # ${MPIEXEC_NUMPROC_FLAG} 2   
	# # ${MPIEXEC_PREFLAGS} --host node001
	# # ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=512  :
	# ${MPIEXEC_NUMPROC_FLAG} 4
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=512 
	# ${MPIEXEC_POSTFLAGS} |sort)
# add_test(Test_Hybrid_HeatFlow_8_processes
	# ${MPIEXEC} 
	# ${MPIEXEC_NUMPROC_FLAG} 4   
	# ${MPIEXEC_PREFLAGS} --host node001
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=512  :
	# ${MPIEXEC_NUMPROC_FLAG} 4
	# ${MPIEXEC_PREFLAGS} --host node002
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hybrid-heatflow --dimx=512 --dimy=512 --dimz=512 
	# ${MPIEXEC_POSTFLAGS} |sort)