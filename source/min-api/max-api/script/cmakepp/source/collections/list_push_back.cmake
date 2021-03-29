# adds a value to the end of the list
function(list_push_back __list_push_back_lst value)
  set(${__list_push_back_lst} ${${__list_push_back_lst}} ${value} PARENT_SCOPE)
endfunction()