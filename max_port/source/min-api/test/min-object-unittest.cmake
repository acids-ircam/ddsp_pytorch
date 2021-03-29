# Copyright 2018 The Min-API Authors. All rights reserved.
# Use of this source code is governed by the MIT License found in the License.md file.

cmake_minimum_required(VERSION 3.10)

set(ORIGINAL_NAME "${PROJECT_NAME}")
set(TEST_NAME "${PROJECT_NAME}_test")
#project(${PROJECT_NAME}_test)

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME}.cpp")

	enable_testing()

	include_directories( 
		"${C74_INCLUDES}"
		"${C74_MIN_API_DIR}/test"
		# "${C74_MIN_API_DIR}/test/mock"
	)

	set(TEST_SOURCE_FILES "")
	FOREACH(SOURCE_FILE ${SOURCE_FILES})
		set(ORIGINAL_WITH_EXT "${ORIGINAL_NAME}.cpp")
		if (SOURCE_FILE STREQUAL ORIGINAL_WITH_EXT)
			set(TEST_SOURCE_FILES ${TEST_SOURCE_FILES} ${TEST_NAME}.cpp)
		else()
			set(TEST_SOURCE_FILES "${TEST_SOURCE_FILES}" ${SOURCE_FILE})
		endif()
	ENDFOREACH()
	
	# set(CMAKE_CXX_FLAGS "-fprofile-arcs -ftest-coverage")
	# set(CMAKE_C_FLAGS "-fprofile-arcs -ftest-coverage")
	# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-arcs -ftest-coverage")

	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../../tests")
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

	if (APPLE)
		#set(CMAKE_OSX_ARCHITECTURES x86_64;i386)
		set(CMAKE_OSX_ARCHITECTURES x86_64)
	endif ()

	add_executable(${TEST_NAME} ${TEST_NAME}.cpp ${TEST_SOURCE_FILES})

	target_compile_definitions(${TEST_NAME} PUBLIC -DMIN_TEST)

	set_property(TARGET ${TEST_NAME} PROPERTY CXX_STANDARD 17)
	set_property(TARGET ${TEST_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

    target_link_libraries(${TEST_NAME} PUBLIC "mock_kernel")

	if (APPLE)
        set_target_properties(${TEST_NAME} PROPERTIES LINK_FLAGS "-Wl,-F'${C74_MAX_API_DIR}/lib/mac', -weak_framework JitterAPI")
		target_compile_options(${TEST_NAME} PRIVATE -DCATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS)
	endif ()
	if (WIN32)
        set_target_properties(${TEST_NAME} PROPERTIES COMPILE_PDB_NAME ${TEST_NAME})

		# target_link_libraries(${TEST_NAME} ${MaxAPI_LIB})
		# target_link_libraries(${TEST_NAME} ${MaxAudio_LIB})
		# target_link_libraries(${TEST_NAME} ${Jitter_LIB})
	endif ()

	add_test(NAME ${TEST_NAME}
	         COMMAND ${TEST_NAME})
	 
endif ()
	 