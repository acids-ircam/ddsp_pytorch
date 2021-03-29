function(val)
  # appends the values to the current_map[current_key]
  stack_peek(:quick_map_map_stack)
  ans(current_map)
  stack_peek(:quick_map_key_stack)
  ans(current_key)
  if(NOT current_map)
    set(res ${ARGN})
    return_ref(res)
  endif()
  map_append("${current_map}" "${current_key}" "${ARGN}")
endfunction()



## val(<val ...>) -> <any...>
##
## adds a val to current property or ref
##
function(val)
  set(args ${ARGN})
  stack_peek(quickmap)
  ans(current_ref)
  
  if(NOT current_ref)
    return()
  endif()
  ## todo check if map 
  address_append("${current_ref}" ${args})
  return_ref(args)
endfunction()
