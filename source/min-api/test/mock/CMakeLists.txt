# Copyright 2018 The Min-API Authors. All rights reserved.
# Use of this source code is governed by the MIT License found in the License.md file.

cmake_minimum_required(VERSION 3.0)

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../min-api-unittest.cmake")
	set(C74_MOCK_TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../tests)
else ()
	set(C74_MOCK_TARGET_DIR .)
endif ()
set(C74_MIN_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../min-api)


#############################################################
project(mock_kernel)
#############################################################


include_directories( 
	"${C74_MIN_API_DIR}/max-api/include"
	"."
)


if (WIN32)
	add_definitions(
		-DWIN_VERSION
		-D_USE_MATH_DEFINES
		-D_CRT_SECURE_NO_WARNINGS
	)
endif ()


if (APPLE)
	#set(CMAKE_OSX_ARCHITECTURES x86_64;i386)
	set(CMAKE_OSX_ARCHITECTURES x86_64)
	set(CMAKE_MACOSX_RPATH 1)
endif ()


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${C74_MOCK_TARGET_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

if (WIN32)
	set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${C74_MOCK_TARGET_DIR}")
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${C74_MOCK_TARGET_DIR}")
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
endif ()

add_library( 
	${PROJECT_NAME} 
	SHARED
	c74_mock.cpp
)


set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
if (WIN32)
#	set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "/ignore:4099")
	set_property(TARGET ${PROJECT_NAME} PROPERTY LINK_FLAGS  "/ignore:4197")
endif()
