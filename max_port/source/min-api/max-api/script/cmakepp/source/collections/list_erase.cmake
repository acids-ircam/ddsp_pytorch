# removes the specified range from lst the start_index is inclusive and end_index is exclusive
#
macro(list_erase __list_erase_lst start_index end_index)
  list_without_range(${__list_erase_lst} ${start_index} ${end_index})
  ans(${__list_erase_lst})
endmacro()