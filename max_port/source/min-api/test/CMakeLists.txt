# Copyright 2020 The Min-API Authors. All rights reserved.
# Use of this source code is governed by the MIT License found in the License.md file.

cmake_minimum_required(VERSION 3.0)

set(C74_MIN_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

enable_testing()

project(min-tests)

# Make our .exe and mock_kernel.dll end up in the same directory
set(OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
set(C74_MOCK_TARGET_DIR ${OUTPUT_DIRECTORY})

set(SOURCES
	atom.cpp
	limit.cpp
	main.cpp
	object.cpp
	symbol.cpp
)

add_executable(min-tests ${SOURCES})

target_compile_definitions(min-tests PUBLIC -DMIN_TEST)

target_include_directories(min-tests PUBLIC
	"${C74_MIN_API_DIR}/include"
	"${C74_MIN_API_DIR}/max-api/include"
	${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(mock)

target_link_libraries(min-tests mock_kernel)

set_target_properties(min-tests PROPERTIES CXX_STANDARD 17)
set_target_properties(min-tests PROPERTIES CXX_STANDARD_REQUIRED ON)

if (APPLE)
	set_target_properties(min-tests PROPERTIES LINK_FLAGS "-Wl,-F'${CMAKE_CURRENT_SOURCE_DIR}/../max-api/lib/mac', -weak_framework JitterAPI")
	target_compile_options(min-tests PRIVATE -DCATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS)
endif ()

add_test(NAME min-tests COMMAND min-tests)
