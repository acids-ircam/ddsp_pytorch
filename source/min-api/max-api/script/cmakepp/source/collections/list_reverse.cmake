## `(<list ref>)-><void>`
##
## reverses the specified lists elements
macro(list_reverse __list_reverse_lst)
  if(${__list_reverse_lst})
    list(REVERSE ${__list_reverse_lst})
  endif()
endmacro()