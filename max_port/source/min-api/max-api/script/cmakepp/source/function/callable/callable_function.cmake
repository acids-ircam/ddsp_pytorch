

## returns the cmake function for the specified callable
function(callable_function input)
  string(MD5  input_key "${input}" )
  get_propertY(callable_func GLOBAL PROPERTY "__global_callable_functions.${input_key}")
  if(NOT callable_func)
    callable("${input}")
    ans(callable)
    get_propertY(callable_func GLOBAL PROPERTY "__global_callable_functions.${input_key}")
  endif()
  set(__ans ${callable_func} PARENT_SCOPE)
endfunction()