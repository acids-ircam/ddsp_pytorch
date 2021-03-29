# convenience function for accessing hg
# use cd() to navigate to working directory
# usage is same as hg command line client
# syntax differs: hg arg1 arg2 ... -> hg(arg1 arg2 ...)
# add a --process-handle flag to get a object containing return code
# input args etc.
# else only console output is returned
function(hg)
    find_package(Hg)
    if (NOT HG_FOUND)
        message(FATAL_ERROR "mercurial is not installed")
    endif ()

    wrap_executable(hg "${HG_EXECUTABLE}")
    hg(${ARGN})
    return_ans()
endfunction()