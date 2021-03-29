## escapes a string to be delimited
## by the the specified delimiters
function(string_encode_delimited str)
    delimiters(${ARGN})
    ans(ds)
    list_pop_front(ds)
    ans(delimiter_begin)
    list_pop_front(ds)
    ans(delimiter_end)

    string(REPLACE \\ \\\\ str "${str}" )
    string(REPLACE "${delimiter_end}" "\\${delimiter_end}" str "${str}" )
    set(str "${delimiter_begin}${str}${delimiter_end}")
    return_ref(str)
endfunction()
