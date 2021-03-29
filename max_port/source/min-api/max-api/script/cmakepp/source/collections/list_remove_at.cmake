# removes all items at all specified indices from list 
function(list_remove_at __list_remove_at_lst)
    if (NOT ${__list_remove_at_lst})
        return()
    endif ()
    set(args)

    foreach (arg ${ARGN})
        list_normalize_index(${__list_remove_at_lst} ${arg})
        ans(res)
        list(APPEND args ${res})
    endforeach ()


    list(REMOVE_AT "${__list_remove_at_lst}" ${args})

    set("${__list_remove_at_lst}" "${${__list_remove_at_lst}}" PARENT_SCOPE)

    return_ref("${__list_remove_at_lst}")

endfunction()