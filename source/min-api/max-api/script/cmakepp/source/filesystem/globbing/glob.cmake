## `(<glob expression...> [--relative] [--recurse]) -> <qualified path...>|<relative path...>`
##
## **flags**:
## * `--relative` causes the output to be paths realtive to current `pwd()`
## * `--recurse` causes the glob expression to be applied recursively
## **scope**
## * `pwd()` influences the relative paths
## **returns**
## * list of files matching the specified glob expressions 
function(glob)
    set(args ${ARGN})
    list_extract_flag(args --relative)
    ans(relative)

    list_extract_flag(args --recurse)
    ans(recurse)

    glob_paths(${args})
    ans(globs)

    if (recurse)
        set(glob_command GLOB_RECURSE)
    else ()
        set(glob_command GLOB)
    endif ()

    if (relative)
        pwd()
        ans(pwd)
        set(relative RELATIVE "${pwd}")
    else ()
        set(relative)
    endif ()

    set(paths)

    if (globs)
        file(${glob_command} paths ${relative} ${globs})
        list_remove_duplicates(paths)
    endif ()

    return_ref(paths)
endfunction()
