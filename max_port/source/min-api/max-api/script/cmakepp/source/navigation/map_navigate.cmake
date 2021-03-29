#navigates a map structure
# use '.' and '[]' operators to select next element in map
# e.g.  map_navigate(<map_ref> res "propa.propb[3].probc[3][4].propd")
function(map_navigate result navigation_expression)
    # path is empty => ""
    if (navigation_expression STREQUAL "")
        return_value("")
    endif ()

    # if navigation expression is a simple var just return it
    if (${navigation_expression})
        return_value(${${navigation_expression}})
    endif ()

    # check for dereference operator
    set(deref false)
    if ("${navigation_expression}" MATCHES "^\\*")
        set(deref true)
        string(SUBSTRING "${navigation_expression}" 1 -1 navigation_expression)
    endif ()

    # split off reference from navigation expression
    unset(ref)
    #_message("${navigation_expression}")
    string(REGEX MATCH "^[^\\[|\\.]*" ref "${navigation_expression}")
    string(LENGTH "${ref}" len)
    string(SUBSTRING "${navigation_expression}" ${len} -1 navigation_expression)


    # if ref is a ref to a ref dereference it :D
    set(not_defined true)
    if (DEFINED ${ref})
        set(ref ${${ref}})
        set(not_defined false)
    endif ()

    # check if ref is valid
    is_address("${ref}")
    ans(is_ref)
    if (NOT is_ref)
        if (not_defined)
            return_value()
        endif ()
        set(${result} "${ref}" PARENT_SCOPE)

        return()
        message(FATAL_ERROR "map_navigate: expected a reference but got '${ref}'")
    endif ()

    # match all navigation expression parts
    string(REGEX MATCHALL "(\\[([0-9][0-9]*)\\])|(\\.[a-zA-Z0-9_\\-][a-zA-Z0-9_\\-]*)" parts "${navigation_expression}")

    # loop through parts and try to navigate
    # if any part of the path is invalid return ""
    set(current "${ref}")
    foreach (part ${parts})
        string(REGEX MATCH "[a-zA-Z0-9_\\-][a-zA-Z0-9_\\-]*" index "${part}")
        string(SUBSTRING "${part}" 0 1 index_type)
        if (index_type STREQUAL ".")
            # get by key
            map_tryget(${current} "${index}")
            ans(current)
        elseif (index_type STREQUAL "[")
            message(FATAL_ERROR "map_navigate: indexation '[<index>]' is not supported")
            # get by index
            address_get(${current})
            ans(lst)
            list(GET lst ${index} keyOrValue)
            map_tryget(${current} ${keyOrValue})
            ans(current)
            if (NOT current)
                set(current "${keyOrValue}")
            endif ()
        endif ()
        if (NOT current)
            return_value("${current}")
        endif ()
    endforeach ()
    if (deref)
        is_address("${current}")
        ans(is_ref)
        if (is_ref)
            address_get("${current}")
            ans(current)
        endif ()
    endif ()
    # current  contains the navigated value
    set(${result} "${current}" PARENT_SCOPE)
endfunction()
	
