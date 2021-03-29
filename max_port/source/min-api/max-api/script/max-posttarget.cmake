# Copyright 2018 The Max-API Authors. All rights reserved.
# Use of this source code is governed by the MIT License found in the License.md file.

if (${C74_CXX_STANDARD} EQUAL 98)
	if (APPLE)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++98 -stdlib=libstdc++")
		set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++")
		set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -stdlib=libstdc++")
	endif ()
else ()
	set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
	set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
endif ()

if ("${PROJECT_NAME}" MATCHES ".*_tilde")
	string(REGEX REPLACE "_tilde" "~" EXTERN_OUTPUT_NAME "${PROJECT_NAME}")
else ()
    set(EXTERN_OUTPUT_NAME "${PROJECT_NAME}")
endif ()
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "${EXTERN_OUTPUT_NAME}")



### Output ###
if (APPLE)
    find_library(JITTER_LIBRARY "JitterAPI" HINTS "${C74_MAX_API_DIR}/lib/mac"  )
    target_link_libraries(${PROJECT_NAME} PUBLIC ${JITTER_LIBRARY})
	
	set_property(TARGET ${PROJECT_NAME}
				 PROPERTY BUNDLE True)
	set_property(TARGET ${PROJECT_NAME}
				 PROPERTY BUNDLE_EXTENSION "mxo")	
	set_target_properties(${PROJECT_NAME} PROPERTIES XCODE_ATTRIBUTE_WRAPPER_EXTENSION "mxo")
	set_target_properties(${PROJECT_NAME} PROPERTIES MACOSX_BUNDLE_BUNDLE_VERSION "${GIT_VERSION_TAG}")
    set_target_properties(${PROJECT_NAME} PROPERTIES MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_LIST_DIR}/Info.plist.in)
elseif (WIN32)
    if ("${PROJECT_NAME}" MATCHES "_test")
    else ()

		target_link_libraries(${PROJECT_NAME} PUBLIC ${MaxAPI_LIB})
		target_link_libraries(${PROJECT_NAME} PUBLIC ${MaxAudio_LIB})
		target_link_libraries(${PROJECT_NAME} PUBLIC ${Jitter_LIB})
	endif ()
	
	if (CMAKE_SIZEOF_VOID_P EQUAL 8)
		set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mxe64")
	else ()
		set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mxe")
	endif ()

	# warning about constexpr not being const in c++14
	set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "/wd4814")

	# do not generate ILK files
	set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "/INCREMENTAL:NO")

	if (EXCLUDE_FROM_COLLECTIVES STREQUAL "yes")
		target_compile_definitions(${PROJECT_NAME} PRIVATE "-DEXCLUDE_FROM_COLLECTIVES")
	endif()

	if (ADD_VERINFO)
		target_sources(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/verinfo.rc)
	endif()
endif ()


### Post Build ###

if (APPLE)
    if ("${PROJECT_NAME}" MATCHES "_test")
    else ()
    	add_custom_command( 
    		TARGET ${PROJECT_NAME} 
    		POST_BUILD 
    		COMMAND cp "${CMAKE_CURRENT_LIST_DIR}/PkgInfo" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${EXTERN_OUTPUT_NAME}.mxo/Contents/PkgInfo" 
    		COMMENT "Copy PkgInfo" 
    	)
    endif ()    
endif ()
