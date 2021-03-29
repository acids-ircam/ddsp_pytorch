## define an inline function
## e.g. `define_function(my_func() message(hello))`
function(define_function function_name)
  arguments_function("${function_name}" 1 ${ARGC})
  return()
endfunction()