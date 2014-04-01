# hetero/samples/mpi/hello_word
# ----------------------------------------------------------------------------
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
	include_directories(${CUDA_INCLUDE_DIRECTORIES})
    message("CUDA is found on this system.")
else(CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif() 
# ----------------------------------------------------------------------------
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)
LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/lib)
# ----------------------------------------------------------------------------
add_executable(cpp-labeling labeling.cpp)

target_link_libraries(cpp-labeling
	hetero_csv
	hetero_cmdparser
	)
# ----------------------------------------------------------------------------
add_test(Test_CPP_Label_Small
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cpp-labeling 
    --inputFile ${RESOURCE_INPUT_PATH}/threshold.raw)
# ----------------------------------------------------------------------------
add_test(Test_CPP_Label_Medium_aa
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cpp-labeling 
	--dimx 4455
	--dimy 3408
	--dimz 128
    --inputFile  ${RESOURCE_INPUT_PATH}/result-mask-7GB-_smallaa
	--outputFile result-mask-7GB-_small.aa)
add_test(Test_CPP_Label_Medium_ab
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cpp-labeling 
	--dimx 4455
	--dimy 3408
	--dimz 128
    --inputFile  ${RESOURCE_INPUT_PATH}/result-mask-7GB-_smallab
	--outputFile result-mask-7GB-_small.ab)
add_test(Test_CPP_Label_Medium_ac
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cpp-labeling 
	--dimx 4455
	--dimy 3408
	--dimz 128
    --inputFile  ${RESOURCE_INPUT_PATH}/result-mask-7GB-_smallac
	--outputFile result-mask-7GB-_small.ac)
	add_test(Test_CPP_Label_Medium_ad
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cpp-labeling 
	--dimx 4455
	--dimy 3408
	--dimz 128
    --inputFile  ${RESOURCE_INPUT_PATH}/result-mask-7GB-_smallad
	--outputFile result-mask-7GB-_small.ad)