message(${CMAKE_CURRENT_SOURCE_DIR})
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
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)
LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/lib)
# ----------------------------------------------------------------------------
add_executable(openmp-for_loop for_loop.cpp)

add_test(Test_OpenMP_For_Loop 
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/openmp-for_loop)