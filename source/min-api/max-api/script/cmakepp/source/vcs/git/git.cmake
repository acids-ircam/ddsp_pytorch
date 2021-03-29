## ''
## wraps the git executable into an easy to use function
function(git)
    find_package(Git)
    if (NOT GIT_FOUND)
        message(FATAL_ERROR "missing git")
    endif ()

    wrap_executable(git "${GIT_EXECUTABLE}")
    git(${ARGN})
    return_ans()
endfunction()  
