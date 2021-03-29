# Copyright 2018 The Min-Lib Authors. All rights reserved.
# Use of this source code is governed by the MIT License found in the License.md file.

cmake_minimum_required(VERSION 3.10)

project(${PROJECT_NAME}_test)

enable_testing()

include_directories( 
	"${C74_INCLUDES}"
	"${CMAKE_CURRENT_LIST_DIR}/../../min-api/test"
#	"${CMAKE_CURRENT_LIST_DIR}/mock"
)

add_definitions(
	-DMIN_TEST
)

#set(CMAKE_CXX_FLAGS "-std=c++1y -stdlib=libc++ -fprofile-arcs -ftest-coverage")
#set(CMAKE_CXX_FLAGS "-fprofile-arcs -ftest-coverage")
#set(CMAKE_C_FLAGS "-fprofile-arcs -ftest-coverage")
#SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-arcs -ftest-coverage")

if (APPLE)
	#set(CMAKE_OSX_ARCHITECTURES x86_64;i386)
	set(CMAKE_OSX_ARCHITECTURES x86_64)
endif ()

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC "mock_kernel")


if (APPLE)
	set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-Wl,-F'${C74_MAX_API_DIR}/lib/mac', -weak_framework JitterAPI")
    target_compile_options(${PROJECT_NAME} PRIVATE -DCATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS)

    # The build dir won't be present the first time the test is compiled.
    # This isn't a problem but it does generate linker warnings about the folder not existing.
    # So we create the folder in advance.

    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Debug")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Release")
endif ()
if (WIN32)
	add_custom_command(	TARGET ${PROJECT_NAME} 
						POST_BUILD	# Adds a post-build event to MyTest
    					COMMAND ${CMAKE_COMMAND} -E copy_if_different		# which executes "cmake - E copy_if_different..."
        				"${CMAKE_CURRENT_SOURCE_DIR}/../../../../tests/mock_kernel.dll"	# <--this is in-file
       				 	${CMAKE_CURRENT_BINARY_DIR}                 					# <--this is out-file path
)
endif ()

add_test(NAME ${PROJECT_NAME}
         COMMAND ${PROJECT_NAME})
