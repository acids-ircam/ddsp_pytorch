function(key key)
  # check if there is a current map
  stack_peek(:quick_map_map_stack)
  ans(current_map)
  if(NOT current_map)
    message(FATAL_ERROR "cannot set key for non existing map be sure to call first map() before first key()")
  endif()
  # set current key
  stack_pop(:quick_map_key_stack)
  stack_push(:quick_map_key_stack "${key}")
endfunction()


## key() -> <void>
##
## starts a new property for a map - may only be called
## after key or map
## fails if current ref is not a map
function(key key)
  stack_pop(quickmap)
  ans(current_key)

  string_take_address(current_key)
  ans(current_ref)
 
  #is_map("${current_ref}")
  is_address("${current_ref}")
  ans(ismap)
  if(NOT ismap)
    message(FATAL_ERROR "expected a map before key() call")
  endif()


  map_set("${current_ref}" "${key}" "")
  stack_push(quickmap "${current_ref}.${key}")
  return()
endfunction()
