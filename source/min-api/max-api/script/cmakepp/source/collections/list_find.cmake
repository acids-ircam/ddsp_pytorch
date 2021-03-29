# searchs lst for value and returns the first idx found
# returns -1 if value is not found
function(list_find __list_find_lst value)
    if (NOT ${__list_find_lst})
        return(-1)
    endif ()
    list(FIND ${__list_find_lst} "${value}" idx)
    return_ref(idx)
endfunction()




