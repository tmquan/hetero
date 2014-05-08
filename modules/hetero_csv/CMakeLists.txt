set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Weffc++ -pedantic -std=c++0x")
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
enable_testing()
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
set(CSV_LIB_NAME hetero_csv)
SET(CSV_VERSION_MAJOR "1")
SET(CSV_VERSION_MINOR "0")
SET(CSV_VERSION_PATCH "0")

set(CSV_VERSION_STRING "${CSV_VERSION_MAJOR}.${CSV_VERSION_MINOR}.${CSV_VERSION_PATCH}")
SET(CSV_LIB_SOVERSION ${CSV_VERSION_MAJOR})
# ----------------------------------------------------------------------------

add_library(${CSV_LIB_NAME} SHARED   
	hetero_csv.cpp)

# # ----------------------------------------------------------------------------  
install(TARGETS ${CSV_LIB_NAME}
  RUNTIME       DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  LIBRARY       DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
  ARCHIVE       DESTINATION share
  COMPONENT library
	)

# install(FILES  hetero_csv.hpp
	# DESTINATION include
	# )
file(COPY hetero_csv.hpp 
	DESTINATION ${CMAKE_HEADER_INCLUDE_DIRECTORY}) 
# ----------------------------------------------------------------------------
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)
LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/lib)
ADD_EXECUTABLE(test_hetero_csv 
	${CMAKE_ACTIVE_TESTING_DIRECTORY}/hetero_csv/test_hetero_csv.cpp)
TARGET_LINK_LIBRARIES(test_hetero_csv ${CSV_LIB_NAME})  
# ----------------------------------------------------------------------------
add_test(Test_CPP_CSV_Parser
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_hetero_csv
    ${RESOURCE_INPUT_PATH}/kiwi.csv)
	