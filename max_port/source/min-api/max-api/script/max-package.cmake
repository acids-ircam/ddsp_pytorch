# Copyright 2018 The Max-API Authors. All rights reserved.
# Use of this source code is governed by the MIT License found in the License.md file.

string(REGEX REPLACE "(.*)/" "" THIS_FOLDER_NAME "${CMAKE_CURRENT_SOURCE_DIR}")
project(${THIS_FOLDER_NAME})


# Set version variables based on the current Git tag
include("${CMAKE_CURRENT_LIST_DIR}/git-rev.cmake")

set(ADD_VERINFO YES)

# Update package-info.json, if present
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/package-info.json.in")
	message("Building _____ ${GIT_TAG} _____")
	set(C74_PACKAGE_NAME "${THIS_FOLDER_NAME}")
	configure_file("${CMAKE_CURRENT_SOURCE_DIR}/package-info.json.in" "${CMAKE_CURRENT_SOURCE_DIR}/package-info.json" @ONLY)

	message("Reading ${CMAKE_CURRENT_SOURCE_DIR}/package-info.json")
	include("${CMAKE_CURRENT_LIST_DIR}/cmakepp/cmakepp.cmake")
	
	file(READ "${CMAKE_CURRENT_SOURCE_DIR}/package-info.json" PKGINFOFILE)
	json_deserialize("${PKGINFOFILE}")
	ans(res)
	
	map_keys("${res}")
	ans(keys)
	
	set(has_reverse_domain false)
	foreach (key ${keys})
		if (key STREQUAL "author")
			nav(res.author)
			ans(AUTHOR)
		endif()
		if (key STREQUAL "package_extra")
			nav(res.package_extra)
			ans(extra)
			map_keys("${extra}")
			ans(extra_keys)						
			foreach (extra_key ${extra_keys})
				if (extra_key STREQUAL "reverse_domain")
					nav(extra.reverse_domain)
					ans(AUTHOR_DOMAIN)			
				endif()
				if (extra_key STREQUAL "copyright")
					nav(extra.copyright)
					ans(COPYRIGHT_STRING)			
				endif()
				if (extra_key STREQUAL "add_verinfo")
					nav(extra.add_verinfo)
					ans(ADD_VERINFO_USER_VALUE)
					if (NOT ADD_VERINFO_USER_VALUE STREQUAL true)
						set(ADD_VERINFO NO)
					endif()
				endif()
				if (extra_key STREQUAL "exclude_from_collectives")
					nav(extra.exclude_from_collectives)
					ans(EXCLUDE_FROM_COLLECTIVES)
				endif()
			endforeach ()			
		endif ()
	endforeach ()
endif ()


# Copy PkgInfo and update Info.plist files on the Mac
if (APPLE)
	message("Generating Info.plist")

	if (NOT DEFINED AUTHOR_DOMAIN)
		set(AUTHOR_DOMAIN "com.acme")
	endif ()
	if (NOT DEFINED COPYRIGHT_STRING)
		set(COPYRIGHT_STRING "Copyright (c) 1974 Acme Inc")
	endif ()
	if (NOT DEFINED EXCLUDE_FROM_COLLECTIVES)
		set(EXCLUDE_FROM_COLLECTIVES "no")
	endif()
	
	set(BUNDLE_IDENTIFIER "\${PRODUCT_NAME:rfc1034identifier}")
	configure_file("${CMAKE_CURRENT_LIST_DIR}/Info.plist.in" "${CMAKE_CURRENT_LIST_DIR}/Info.plist" @ONLY)
endif ()

if (WIN32 AND ADD_VERINFO)
	message("Generating verinfo.rc")

	if (NOT DEFINED AUTHOR_DOMAIN)
		set(AUTHOR_DOMAIN "com.acme")
	endif ()
	if (NOT DEFINED COPYRIGHT_STRING)
		set(COPYRIGHT_STRING "Copyright (c) 1974 Acme Inc")
	endif ()
	if (NOT DEFINED EXCLUDE_FROM_COLLECTIVES)
		set(EXCLUDE_FROM_COLLECTIVES "no")
	endif()

	configure_file("${CMAKE_CURRENT_LIST_DIR}/verinfo.rc.in" "${CMAKE_CURRENT_LIST_DIR}/verinfo.rc" @ONLY)
endif()

# Macro from http://stackoverflow.com/questions/7787823/cmake-how-to-get-the-name-of-all-subdirectories-of-a-directory
MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
        LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()

