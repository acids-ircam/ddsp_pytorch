function(stack_push stack)
  map_tryget("${stack}" back)
  ans(current_index)
  
  # increase stack pointer
  if(NOT current_index)
    set(current_index 0)
  endif()
  math(EXPR current_index "${current_index} + 1")
  map_set_hidden("${stack}" back "${current_index}")

  map_set_hidden("${stack}" "${current_index}" "${ARGN}")
endfunction()
