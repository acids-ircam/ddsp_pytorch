## checks of the handler can handle the specified request
## this is done by look at the first input argument and checking if
## it is contained in labels
function(handler_match handler request)
    map_tryget(${handler} labels)
    ans(labels)

    map_tryget(${request} input)
    ans(input)

    list_pop_front(input)
    ans(cmd)

    list_contains(labels "${cmd}")
    ans(is_match)

    return_ref(is_match)
endfunction()