function(map_dfs_references_once)
    if(NOT __recurse)
        set(__recurse true)
        map_new()
        ans(visited)
    endif()


    list(LENGTH ARGN count)
    if("${count}" EQUAL 0)
        return(${visited})
    endif()


    if("${count}" GREATER "1")
        foreach(arg ${ARGN})
            map_dfs_references_once(${arg})
        endforeach()
        return(${visited})
    endif()

    is_map("${ARGN}")
    ans(is_map)

    if(NOT is_map)
        return(${visited})
    endif()
    map_tryget(${visited} "${ARGN}")
    ans(result)

    if(result)
        return(${visited})
    endif()


    map_set(${visited} "${ARGN}" "true")


    map_keys("${ARGN}")
    ans(keys)

    foreach(key ${keys})
        map_tryget("${ARGN}" "${key}")
        ans(keyVal)
        map_dfs_references_once("${keyVal}")
    endforeach()

    return(${visited})

endfunction()