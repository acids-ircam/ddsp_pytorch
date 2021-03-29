# returns a list containing all elements contained
# in all passed list references
function(list_intersect)
    set(__list_intersect_lists ${ARGN})

    list(LENGTH __list_intersect_lists __list_intersect_lists_length)
    if (NOT __list_intersect_lists_length)
        return()
    endif ()

    if ("${__list_intersect_lists_length}" EQUAL 1)
        if (${__list_intersect_lists})
            list(REMOVE_DUPLICATES "${__list_intersect_lists}")
        endif ()
        return_ref("${__list_intersect_lists}")
    endif ()


    list_pop_front(__list_intersect_lists)
    ans(__list_intersect_first)
    list_intersect(${__list_intersect_first})
    ans(__list_intersect_current_elements)
    # __list_intersect_current_elements is now unique

    # intersect rest elements
    list_intersect(${__list_intersect_lists})
    ans(__list_intersect_rest_elements)

    # get elements which are to be removed from list
    set(__list_intersect_elements_to_remove ${__list_intersect_current_elements})
    if (__list_intersect_elements_to_remove)
        foreach (__list_operation_item ${__list_intersect_rest_elements})
            list(REMOVE_ITEM __list_intersect_elements_to_remove ${__list_operation_item})
        endforeach ()
    endif ()
    # remove elements and return result
    if (__list_intersect_elements_to_remove)
        list(REMOVE_ITEM __list_intersect_current_elements ${__list_intersect_elements_to_remove})
    endif ()
    return_ref(__list_intersect_current_elements)
endfunction()