set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Weffc++ -pedantic -std=c++0x")
# ----------------------------------------------------------------------------
# find_package(Boost COMPONENTS program_options REQUIRED)
# if(Boost_FOUND)
    # message("Boost is found on this system.")
	# include_directories(${Boost_INCLUDE_DIRS})
    # link_directories(${Boost_LIBRARY_DIRS})
# else(Boost_FOUND)
    # message(FATAL_ERROR "Boost is not installed on this system.")
# endif()
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
SET(CSV_VERSION_MAJOR "0")
SET(CSV_VERSION_MINOR "1")
SET(CSV_VERSION_PATCH "0")

set(CSV_VERSION_STRING "${CSV_VERSION_MAJOR}.${CSV_VERSION_MINOR}.${CSV_VERSION_PATCH}")
SET(CSV_LIB_SOVERSION ${CSV_VERSION_MAJOR})


# Name of C++ library
set(DATA_LIB_NAME hetero_data)

set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)
# ----------------------------------------------------------------------------
set(DATA_HEADERS_DIR ${CMAKE_DEFINE_MODULES_DIRECTORY}/hetero_data)

set(CSV_HPP  ${DATA_HEADERS_DIR}/hetero_csv.hpp)
set(CSV_CPP  ${DATA_HEADERS_DIR}/hetero_csv.cpp)

set(DATA_HPP ${CSV_HPP} "${DATA_HEADERS_DIR}/hetero_data.hpp")
source_group("Header Files" FILES ${CSV_HPP})
source_group("Source Files" FILES ${CSV_CPP})


# include_directories(${CSV_HEADERS_DIR})

set(SOURCES ${CSV_HPP} ${CSV_CPP}
    ${DATA_HPP})

add_library(${DATA_LIB_NAME} SHARED  ${DATA_HPP}
    ${CSV_CPP})

  
install(TARGETS ${DATA_LIB_NAME}
  RUNTIME       DESTINATION bin
  LIBRARY       DESTINATION lib
  ARCHIVE       DESTINATION share 
  PUBLIC_HEADER DESTINATION include
  COMPONENT library
)
file(COPY ${DATA_HPP} DESTINATION ${CMAKE_HEADER_INCLUDE_DIRECTORY}) 
# install(FILES ${DATA_HPP} 
    # DESTINATION ${CMAKE_HEADER_INCLUDE_DIRECTORY}) 
      
# ----------------------------------------------------------------------------
# Merge the libraries
# set(HETERO_DATA  ${DATA_LIB_NAME})
# ----------------------------------------------------------------------------
enable_testing()
SET(TEST_SRC     ${CMAKE_ACTIVE_TESTING_DIRECTORY}/hetero_data/test_hetero_csv.cpp)

LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/lib)
ADD_EXECUTABLE(test_hetero_csv ${TEST_SRC})
TARGET_LINK_LIBRARIES(test_hetero_csv ${DATA_LIB_NAME})  
# ----------------------------------------------------------------------------
add_test(Test_CPP_DSV_Parser
    ${RUNTIME_OUTPUT_DIRECTORY}/test_hetero_csv
    ${RESOURCE_INPUT_PATH}/supernova.csv)