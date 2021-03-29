function(test)
  message("Test Inconclusive")
  return()
  # {
  #   name:<string>,
  #   author:<single line string>,
  #   copyright_range: [<begin:<year>> [<end:<year>>]]
  #   license_name:''
  #   license_url:''
  #   license:<multi line string>,
  #   cmake_function_files:<glob>  // are globbed and appended in 
  #   
  # }
#http://www.cmake.org/cmake/help/v3.0/manual/cmake-developer.7.html#find-modules
  function(listing_append_comment listing comment)

    string_split("${comment}" "\n")
    ans(comment)
    foreach(line ${comment})
      list(APPEND "#${line}")
    endforeach()
    set(listing ${listing})
  endfunction()

  function(cmake_module_create config)
    obj("${config}")
    ans(config)


    ## imports date vars yyyy MM dd 
    datetime()
    ans(dt)
    scope_import_map(${dt})


    # imports module config object    
    scope_import_map("${config}")



    list_pop_front(copyright_range)
    ans(copyright_begin)
    list_pop_front(copyright_range)
    ans(copyright_end)
    if(NOT copyright_begin)
      set(copyright_begin ${yyyy})
    endif()
    if(NOT copyright_end)
      set(copyright_end ${yyyy})
    endif()

    set(listing)

    list(APPEND listing "#=============================================================================")
    list(APPEND listing "# Copyright ${copyright_begin}-${copyright_end} ${author}")
    list(APPEND listing "#")
    list(APPEND listing "# Distributed under the OSI-approved BSD License (the \"License\")")
    list(APPEND listing "# see accompanying file Copyright.txt for details.")
    list(APPEND listing "#")
    list(APPEND listing "# This software is distributed WITHOUT ANY WARRANTY without even the")
    list(APPEND listing "# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.")
    list(APPEND listing "# See the License for more information.")
    list(APPEND listing "#=============================================================================")
    list(APPEND listing "# (To distribute this file outside of CMake, substitute the full")
    list(APPEND listing "#  License text for the above reference.)")
    list(APPEND listing "")


    list(APPEND listing "set(${name}_FOUND true PARENT_SCOPE)")

    # find the package base dir

    #list(APPEND "find_package()")
    source_list_compile("${listing}")
    return_ans()
  endfunction()

  function(cmake_module_write tgt config)
    obj("${config}")
    ans(config)
    cmake_module_create("${config}")
    ans(res)
    path("${tgt}")
    ans(tgt)


    if(IS_DIRECTORY "${tgt}")
      map_tryget(${config} name)
      ans(name)
      set(tgt "${tgt}/Find${name}.cmake")

    endif()
    fwrite("${tgt}" "${res}")
    return_ref(tgt)
  endfunction()


  cmake_module_write(. "
  {
    name:'TestModule'
  }
  ")


  assert(EXISTS "${test_dir}/FindTestModule.cmake")


  set(CMAKE_MODULE_PATH "${test_dir}")  
  find_package(TestModule)
  assert(TestModule_FOUND)


  set(TESTMODULE_FOUND)
  set(CMAKE_MODULE_PATH)
  set(TestModule_DIR "${test_dir}")
  find_package(TestModule)
  assert(TestModule_FOUND)


# input vars

#Xxx_FIND_QUIETLY # do not complain
#Xxx_FIND_REQUIRED #use fatal errror if not found

#Xxx_FIND_COMPONENTS
#Xxx_FIND_REQUIRED_Yyy


#Xxx_FIND_VERSION
#Xxx_FIND_VERSION_MAJOR
#Xxx_FIND_VERSION_MINOR
#Xxx_FIND_VERSION_PATCH
#Xxx_FIND_VERSION_TWEAK
#Xxx_FIND_VERSION_COUNT
#Xxx_FIND_VERSION_EXACT

#Xxx_INCLUDE_DIRS
#Xxx_LIBRARIES
#Xxx_DEFINITIONS
#Xxx_EXECUTABLE
#Xxx_Yyy_EXECUTABLE
#Xxx_LIBRARY_DIRS
#Xxx_ROOT_DIR
#Xxx_WRAP_Yy
#Xxx_VERSION_Yy
#Xxx_Yy_FOUND
#Xxx_FOUND
#Xxx_NOT_FOUND_MESSAGE
#Xxx_RUNTIME_LIBRARY_DIRS
#Xxx_VERSION_STRING
#Xxx_VERSION_MAJOR
#Xxx_VERSION_MINOR
#Xxx_VERSION_PATCH
#Xxx_LIBRARY
#Xxx_Yy_LIBRARY
#Xxx_INCLUDE_DIR
#Xxx_Yy_INCLUDE_DIR

#.rst:
# FindJPEG
# --------
#
# Find JPEG
#
# Find the native JPEG includes and library This module defines
#
# ::
#
#   JPEG_INCLUDE_DIR, where to find jpeglib.h, etc.
#   JPEG_LIBRARIES, the libraries needed to use JPEG.
#   JPEG_FOUND, If false, do not try to use JPEG.
#
# also defined, but not for general use are
#
# ::
#
#   JPEG_LIBRARY, where to find the JPEG library.

#=============================================================================
# Copyright 2001-2009 Kitware, Inc.
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)



#find_path(JPEG_INCLUDE_DIR jpeglib.h)

#set(JPEG_NAMES ${JPEG_NAMES} jpeg)
#find_library(JPEG_LIBRARY NAMES ${JPEG_NAMES} )

# handle the QUIETLY and REQUIRED arguments and set JPEG_FOUND to TRUE if
# all listed variables are TRUE
#include(${CMAKE_CURRENT_LIST_DIR}/FindPackageHandleStandardArgs.cmake)
#FIND_PACKAGE_HANDLE_STANDARD_ARGS(JPEG DEFAULT_MSG JPEG_LIBRARY JPEG_INCLUDE_DIR)

#if(JPEG_FOUND)
#  set(JPEG_LIBRARIES ${JPEG_LIBRARY})
#endif()

# Deprecated declarations.
#set (NATIVE_JPEG_INCLUDE_PATH ${JPEG_INCLUDE_DIR} )
#if(JPEG_LIBRARY)
#  get_filename_component (NATIVE_JPEG_LIB_PATH ${JPEG_LIBRARY} PATH)
#endif()

#mark_as_advanced(JPEG_LIBRARY JPEG_INCLUDE_DIR )


endfunction()