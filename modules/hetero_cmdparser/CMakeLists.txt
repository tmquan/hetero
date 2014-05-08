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
set(CMDPARSER_LIB_NAME hetero_cmdparser)
SET(CMDPARSER_VERSION_MAJOR "1")
SET(CMDPARSER_VERSION_MINOR "0")
SET(CMDPARSER_VERSION_PATCH "0")

set(CMDPARSER_VERSION_STRING "${CMDPARSER_VERSION_MAJOR}.${CMDPARSER_VERSION_MINOR}.${CMDPARSER_VERSION_PATCH}")
SET(CMDPARSER_LIB_SOVERSION ${CMDPARSER_VERSION_MAJOR})
# ----------------------------------------------------------------------------

add_library(${CMDPARSER_LIB_NAME} SHARED   
	hetero_cmdparser.cpp)

# # ----------------------------------------------------------------------------  
install(TARGETS ${CMDPARSER_LIB_NAME}
  RUNTIME       DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  LIBRARY       DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
  ARCHIVE       DESTINATION share
  COMPONENT library
	)

# install(FILES  hetero_cmdparser.hpp
	# DESTINATION ${CMAKE_HEADER_INCLUDE_DIRECTORY}
	# )
file(COPY hetero_cmdparser.hpp 
	DESTINATION ${CMAKE_HEADER_INCLUDE_DIRECTORY}) 
# # ----------------------------------------------------------------------------
# ADD_EXECUTABLE(test_hetero_cmdparser 
	# ${CMAKE_ACTIVE_TESTING_DIRECTORY}/hetero_cmdparser/test_hetero_cmdparser.cpp)
# TARGET_LINK_LIBRARIES(test_hetero_cmdparser ${CMDPARSER_LIB_NAME})  
# # ----------------------------------------------------------------------------
# add_test(Test_CPP_CMDPARSER_Parser
    # ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_hetero_cmdparser
    # ${RESOURCE_INPUT_PATH}/kiwi.csv)
	