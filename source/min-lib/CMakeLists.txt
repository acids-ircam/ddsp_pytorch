# Copyright 2018 The Min-Lib Authors. All rights reserved.
# Use of this source code is governed by the MIT License found in the License.md file.

cmake_minimum_required(VERSION 3.10)
project(MinLib)


if (${CMAKE_GENERATOR} MATCHES "Xcode")
 		if (${XCODE_VERSION} VERSION_LESS 10)
  			message(STATUS "Xcode 10 is required. Please install from the Mac App Store.")
			return ()
		endif ()
endif ()


#include(${CMAKE_CURRENT_SOURCE_DIR}/script/min-package.cmake)


file(GLOB_RECURSE MIN_LIB_HEADERS
	 RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
	 ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
add_custom_target( LIB ALL
	SOURCES ${MIN_LIB_HEADERS}
)


# Add unit tests for the Lib
if ("${THIS_PACKAGE_NAME}" MATCHES ".*devkit")

	enable_testing()

	SUBDIRLIST(TESTDIRS ${CMAKE_CURRENT_SOURCE_DIR}/test)
	foreach(testdir ${TESTDIRS})
			if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/${testdir}/CMakeLists.txt")
				message("Generating Unit Test: ${testdir}")
				add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test/${testdir})
			endif ()
	endforeach()

endif ()
