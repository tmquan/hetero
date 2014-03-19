set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Weffc++ -pedantic -std=c++0x")
# ----------------------------------------------------------------------------
find_package(Boost COMPONENTS program_options REQUIRED)
if(Boost_FOUND)
    message("Boost is found on this system.")
	include_directories(${Boost_INCLUDE_DIRS})
else(Boost_FOUND)
    message(FATAL_ERROR "Boost is not installed on this system.")
endif()
# ----------------------------------------------------------------------------
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
	message("OpenMP is found on this system.")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else(OPENMP_FOUND)
    message(FATAL_ERROR "OpenMP is not installed on this system.")
endif()
# ----------------------------------------------------------------------------
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
	include_directories(${CUDA_INCLUDE_DIRECTORIES})
    message("CUDA is found on this system.")
else(CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif() 
# ----------------------------------------------------------------------------
cuda_add_executable(cpp-split_data 
					split_data.cpp
					# split_data.cu
					cmdparser.cpp
					datparser.cpp
			   )
target_link_libraries(cpp-split_data ${Boost_PROGRAM_OPTIONS_LIBRARY} ${Boost_REGEX_LIBRARY} )
# target_link_libraries(cpp-split_data ${Boost_LIBRARIES} )
target_link_libraries(cpp-split_data ${OpenMP_CXX_FLAGS} )
# ----------------------------------------------------------------------------
# add_test(Test_CPP_Split_Data
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cpp-split_data
	# # --input	${RESOURCE_INPUT_PATH}/em_medium.dat
	# --inputFile		${RESOURCE_INPUT_PATH}/em_small.dat 
	# --outputDir     ${RESOURCE_INPUT_PATH}/
	# --numx  2
	# --numy 	2
	# --numz 	2   
	# --halo	1
	# ) 
add_test(Test_CPP_Split_Data
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cpp-split_data
	# --input	${RESOURCE_INPUT_PATH}/em_medium.dat
	--inputFile		${RESOURCE_INPUT_PATH}/em_medium.dat 
	--outputDir     ${RESOURCE_INPUT_PATH}/
	--numx  1 	
	--numy 	1
	--numz 	32   
	--halo	13
	) 