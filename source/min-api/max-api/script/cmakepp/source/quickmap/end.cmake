function(end)
  # remove last key from key stack and last map from map stack
  # return the popped map
  stack_pop(:quick_map_key_stack)
  stack_pop(:quick_map_map_stack)
  return_ans()
endfunction()



## end() -> <current value>
##
## ends the current key, ref or map and returns the value
## 
function(end)
  stack_pop(quickmap)
  ans(ref)

  if(NOT ref)
    message(FATAL_ERROR "end() not possible ")
  endif()
    
  string_take_address(ref)
  ans(current_ref)

  return_ref(current_ref)
endfunction()
