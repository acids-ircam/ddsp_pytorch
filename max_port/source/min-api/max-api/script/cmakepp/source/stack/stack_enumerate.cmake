

# returns all elements of the stack possibly fucking up
# element count because single elements may be lists-
# -> lists are flattened
function(stack_enumerate stack)
  map_tryget("${stack}" back)
  ans(current_index)
  if(NOT current_index)
    return()
  endif()
  
 # math(EXPR current_index "${current_index} - 1")
  set(res)
  foreach(i RANGE 1 ${current_index})
    map_tryget("${stack}" "${i}")
    ans(current)
    list(APPEND res "${current}")
  endforeach()
  return_ref(res)
endfunction()