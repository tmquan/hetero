# --------------------------------------------------------------
# Top level Hetero project
# Author		:	Tran Minh Quan
# Email			: 	quantm@unist.ac.kr
# 					quantm@kaist.ac.kr
# Address		: 	High-performance Visual Computing Lab
# Website		:	http://hvcl.unist.ac.kr/quantm/
# --------------------------------------------------------------
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
# --------------------------------------------------------------
project(Hetero CXX C)

# ----------------------------------------------------------------------------
# Process subdirectories
# ----------------------------------------------------------------------------
# hetero.hpp and legacy headers
add_subdirectory(include)

# Hetero modules
add_subdirectory(modules)

# Generate targets for documentation
add_subdirectory(doc)

# various data that is used by hetero libraries and/or demo applications.
add_subdirectory(data)

# examples
add_subdirectory(samples)