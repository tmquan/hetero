# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Weffc++ -pedantic -std=c++0x")
# ----------------------------------------------------------------------------
find_package(Boost COMPONENTS program_options REQUIRED)
if(Boost_FOUND)
    message("Boost is found on this system.")
	include_directories(${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
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
# SET(CSV_VERSION_MAJOR "0")
# SET(CSV_VERSION_MINOR "1")
# SET(CSV_VERSION_PATCH "0")

# set(CSV_VERSION_STRING "${CSV_VERSION_MAJOR}.${CSV_VERSION_MINOR}.${CSV_VERSION_PATCH}")
# SET(CSV_LIB_SOVERSION ${CSV_VERSION_MAJOR})


# # Name of C++ library
# set(CSV_LIB_NAME hetero_csv)

# set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)
# # ----------------------------------------------------------------------------
# set(CSV_HEADERS_DIR ${CMAKE_DEFINE_MODULES_DIRECTORY}/hetero_data)
# # set(CSV_HEADERS_DIR ${CMAKE_HEADER_INCLUDE_DIRECTORY})

# set(CSV_HPP
  # ${CSV_HEADERS_DIR}/hetero_csv.hpp
# )

# set(CSV_CPP
  # ${CSV_HEADERS_DIR}/hetero_csv.cpp
# )

# source_group("Header Files" FILES ${CSV_HPP})
# source_group("Source Files" FILES ${CSV_CPP})

# include_directories(../include)

# set(SOURCES
  # ${CSV_HPP}
  # ${CSV_CPP})

# add_library(${CSV_LIB_NAME} SHARED ${SOURCES})

# target_link_libraries(${CSV_LIB_NAME} ${Boost_LIBRARIES} 
    # ${Boost_PROGRAM_OPTIONS_LIBRARY} ${Boost_REGEX_LIBRARY} )
# set_target_properties(${CSV_LIB_NAME}
  # PROPERTIES SOVERSION "${CSV_LIB_SOVERSION}" )

# if (APPLE)
  # set_target_properties(
    # ${CSV_LIB_NAME}
    # PROPERTIES
    # INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")
# endif()

# install(TARGETS ${CSV_LIB_NAME}
  # RUNTIME DESTINATION bin
  # LIBRARY DESTINATION lib
  # ARCHIVE DESTINATION share)

# install(DIRECTORY ${CMAKE_HEADER_INCLUDE_DIRECTORY}
  # DESTINATION .
  # FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# enable_testing()

# INCLUDE_DIRECTORIES(
    # ./include
    # )
# SET(UNIT_TEST test_hetero_csv)


# SET(TEST_SRC
    # ${CMAKE_ACTIVE_TESTING_DIRECTORY}/hetero_data/main.cpp
    # )
# add_definitions("-DBOOST_TEST_DYN_LINK")
# add_definitions("-DBoost_NO_BOOST_CMAKE=ON")
# add_definitions("-DBOOST_ALL_NO_LIB")


# # set(BUILD_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# # Output directory in which to build RUNTIME target files.
# # set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_OUTPUT_DIRECTORY})

# ADD_EXECUTABLE(${UNIT_TEST} ${TEST_SRC})

# TARGET_LINK_LIBRARIES(${UNIT_TEST} ${CSV_LIB_NAME}  
    # ${Boost_PROGRAM_OPTIONS_LIBRARY} ${Boost_REGEX_LIBRARY} 
    # ${Boost_LIBRARIES} )

# ADD_TEST(${UNIT_TEST} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${UNIT_TEST} ${PROJECT_SOURCE_DIR}/test/data)
# cuda_add_executable(cpp-dsv 
					# dsv.cpp
			   # )
 # target_link_libraries               
# target_link_libraries(cpp-dsv ${Boost_PROGRAM_OPTIONS_LIBRARY} ${Boost_REGEX_LIBRARY} )
# # target_link_libraries(cpp-dsv ${Boost_LIBRARIES} )
# target_link_libraries(cpp-dsv ${OpenMP_CXX_FLAGS} )
# target_link_libraries(cpp-dsv)
# ----------------------------------------------------------------------------
# add_test(Test_CPP_DSV_Parser
	# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cpp-dsv
	# # --inputCSVFile		${RESOURCE_INPUT_PATH}/supernova.dsv
	# ${RESOURCE_INPUT_PATH}/supernova.dsv , ,
	# ) 