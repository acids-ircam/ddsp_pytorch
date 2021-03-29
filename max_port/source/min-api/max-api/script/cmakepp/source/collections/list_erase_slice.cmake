# removes the specified range from lst and returns the removed elements
macro(list_erase_slice __list_erase_slice_lst start_index end_index)
  list_slice(${__list_erase_slice_lst} ${start_index} ${end_index})
  ans(__res)

  list_without_range(${__list_erase_slice_lst} ${start_index} ${end_index})
  ans(${__list_erase_slice_lst})
  set(__ans ${__res})
  #set(${__list_erase_slice_lst} ${rest} PARENT_SCOPE)
  #return_ref(res)
endmacro()



