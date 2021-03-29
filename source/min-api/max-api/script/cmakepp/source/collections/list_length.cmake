## returns the length of the specified list
macro(list_length __list_count_lst)
    list(LENGTH "${__list_count_lst}" __ans)
endmacro()
