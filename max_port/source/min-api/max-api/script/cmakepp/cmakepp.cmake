## cmakepp 
##
## An enhancement suite for CMake
## 
##
## This file is the entry point for cmakepp. If you want to use the functions 
## just include this file.
##
## it can also be used as a module file with cmake's find_package() 

cmake_minimum_required(VERSION 3.1)

get_property(is_included GLOBAL PROPERTY cmakepp_include_guard)
if (is_included)
    _return()
endif ()
set_property(GLOBAL PROPERTY cmakepp_include_guard true)

if (POLICY CMP0053)
    ## for template compile
    cmake_policy(SET CMP0053 NEW)
endif ()
if (POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif ()
# installation dir of cmakepp
set(cmakepp_base_dir "${CMAKE_CURRENT_LIST_DIR}")

# include functions needed for initializing cmakepp
include(CMakeParseArguments)


# get temp dir which is needed by a couple of functions in cmakepp
# first uses env variable TMP if it does not exists TMPDIR is used
# if both do not exists current_list_dir/tmp is used
if (UNIX)
    set(cmakepp_tmp_dir $ENV{TMPDIR} /tmp)
else ()
    set(cmakepp_tmp_dir $ENV{TMP} ${CMAKE_CURRENT_LIST_DIR}/tmp)
endif ()
list(GET cmakepp_tmp_dir 0 cmakepp_tmp_dir)
file(TO_CMAKE_PATH "${cmakepp_tmp_dir}" cmakepp_tmp_dir)

# dummy function which is overwritten and in this form just returns the temp_dir
function(cmakepp_config key)
    return("${cmakepp_tmp_dir}")
endfunction()

## create invoke later functions 
# function(task_enqueue callable)
#   ## semicolon encode before string_encode_semicolon exists
#   string(ASCII  31 us)
#   string(REPLACE ";" "${us}" callable "${callable}")
#   set_property(GLOBAL APPEND PROPERTY __initial_invoke_later_list "${callable}") 
#   return()
# endfunction()

include("${cmakepp_base_dir}/source/type/parameter_definition.cmake")
include("${cmakepp_base_dir}/source/task/task_enqueue.cmake")

## includes all cmake files of cmakepp 
include("${cmakepp_base_dir}/source/core/require.cmake")
require("${cmakepp_base_dir}/source/*.cmake")

## include task_enqueue last
include("${cmakepp_base_dir}/source/task/task_enqueue.cmake")

## setup global variables to contain command_line_args
parse_command_line(command_line_args "${command_line_args}") # parses quoted command line args
map_set(global "command_line_args" ${command_line_args})
map_set(global "unused_command_line_args" ${command_line_args})

## todo... change this 
# setup cmakepp config
map()
kv(base_dir
        LABELS --cmakepp-base-dir
        MIN 1 MAX 1
        DISPLAY_NAME "cmakepp installation dir"
        DEFAULT "${CMAKE_CURRENT_LIST_DIR}"
        )
kv(keep_temp
        LABELS --keep-tmp --keep-temp -kt
        MIN 0 MAX 0
        DESCRIPTION "does not delete temporary files after")
kv(temp_dir
        LABELS --temp-dir
        MIN 1 MAX 1
        DESCRIPTION "the directory used for temporary files"
        DEFAULT "${cmakepp_tmp_dir}/cutil/temp"
        )
kv(cache_dir
        LABELS --cache-dir
        MIN 1 MAX 1
        DESCRIPTION "the directory used for caching data"
        DEFAULT "${cmakepp_tmp_dir}/cutil/cache"
        )
kv(bin_dir
        LABELS --bin-dir
        MIN 1 MAX 1
        DEFAULT "${CMAKE_CURRENT_LIST_DIR}/bin"
        )
kv(cmakepp_path
        LABELS --cmakepp-path
        MIN 1 MAX 1
        DEFAULT "${CMAKE_CURRENT_LIST_FILE}"
        )

end()
ans(cmakepp_config_definition)

cd("${CMAKE_SOURCE_DIR}")
# setup config_function for cmakepp
config_setup("cmakepp_config" ${cmakepp_config_definition})

## run all currently enqueued tasks
set(cmakepp_is_loaded true)
task_enqueue("[]()") ## dummy
tqr()

## register all function defs
parameter_definition("")

## check if in script mode and script file is equal to this file
## then invoke either cli mode
cmake_entry_point()
ans(entry_point)
if ("${CMAKE_CURRENT_LIST_FILE}" STREQUAL "${entry_point}")
    cmakepp_cli()
endif ()

## variables expected by cmake's find_package method
set(CMAKEPP_FOUND true)

set(CMAKEPP_VERSION_MAJOR "0")
set(CMAKEPP_VERSION_MINOR "0")
set(CMAKEPP_VERSION_PATCH "0")
set(CMAKEPP_VERSION "${CMAKEPP_VERSION_MAJOR}.${CMAKEPP_VERSION_MINOR}.${CMAKEPP_VERSION_PATCH}")
set(CMAKEPP_BASE_DIR "${cmakepp_base_dir}")
set(CMAKEPP_BIN_DIR "${cmakepp_base_dir}/bin")
set(CMAKEPP_TMP_DIR "${cmakepp_tmp_dir}")

set(cmakepp_path "${CMAKE_CURRENT_LIST_FILE}")
set(CMAKEPP_PATH "${CMAKE_CURRENT_LIST_FILE}")
## setup file
set(ENV{CMAKEPP_PATH} "${CMAKE_CURRENT_LIST_FILE}")
