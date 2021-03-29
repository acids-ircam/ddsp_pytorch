## `(<base_dir:<qualified path>> <~path>) -> <qualified path>`
##
## @todo realpath or abspath?
## qualfies a path using the specified base_dir
##
## if path is absolute (starts with / or under windows with <drive letter>:/) 
## it is returned as is
##
## if path starts with a '~' (tilde) the path is 
## qualfied by prepending the current home directory (on all OSs)
##
## is neither absolute nor starts with ~
## the path is relative and it is qualified 
## by prepending the specified <base dir>
function(path_qualify_from base_dir path)
    string(REPLACE \\ / path "${path}")
    get_filename_component(realpath "${path}" ABSOLUTE)

    ## windows absolute path
    if (WIN32 AND "_${path}" MATCHES "^_[a-zA-Z]:\\/")
        return_ref(realpath)
    endif ()

    ## posix absolute path
    if ("_${path}" MATCHES "^_\\/")
        return_ref(realpath)
    endif ()


    ## home path
    if ("_${path}" MATCHES "^_~\\/?(.*)")
        home_dir()
        ans(base_dir)
        set(path "${CMAKE_MATCH_1}")
    endif ()

    set(path "${base_dir}/${path}")

    ## relative path
    get_filename_component(realpath "${path}" ABSOLUTE)

    return_ref(realpath)
endfunction()