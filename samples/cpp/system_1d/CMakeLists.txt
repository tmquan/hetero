message(${CMAKE_CURRENT_SOURCE_DIR})
# ----------------------------------------------------------------------------
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)
LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/lib)
# ----------------------------------------------------------------------------
add_executable(cpp-system_1d system_1d.cpp)

add_test(Test_CPP_System_1D
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cpp-system_1d
	${RESOURCE_INPUT_PATH}/garnet.unist.ac.kr
	)