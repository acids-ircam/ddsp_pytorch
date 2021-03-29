function(cmake_token_range_variable range var_name)
  set(args ${ARGN})

  cmake_invocation_filter_token_range("${range}" 
    invocation_identifier STREQUAL "set" 
    AND invocation_arguments MATCHES "^${var_name}"  
    --take 1 
    )
  ans(invocation)

  
  if(NOT invocation)
    messagE(FATAL_ERROR "could not find 'set(${var_name} ...)'")
  endif()

  map_tryget(${invocation} invocation_token)
  ans(invocation_token)
  map_tryget(${invocation} invocation_arguments)
  ans(arguments)
  list_pop_front(arguments) ## remove var_name
  list_modify(arguments ${args})
  cmake_invocation_token_set_arguments(${invocation_token} ${var_name} ${arguments})
  return_ref(arguments)
endfunction()
