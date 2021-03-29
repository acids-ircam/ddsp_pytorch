# adds a value at the beginning of the list
function(list_push_front __list_push_front_lst value)
  set(${__list_push_front_lst} ${value} ${${__list_push_front_lst}} PARENT_SCOPE)   
  return(true)
endfunction()