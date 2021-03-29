
function(callable_call callable)
  map_get_special("${callable}" callable_function)
  eval("${__ans}(${ARGN})")
  set(__ans "${__ans}" PARENT_SCOPE)
endfunction()
