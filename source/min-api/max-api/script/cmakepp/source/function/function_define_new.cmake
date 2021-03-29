## `(<signarture> <cmake code>)-><function name>`
##
## defines an anonymous cmake function and returns its reference (name)
## the code needs to begin with the signature without the name e.g. `(arg1 arg2)` 
##  
function(function_define_new)
  arguments_cmake_code(0 ${ARGC})
  ans(code)
  anonymous_function_new("${code}")
  return_ans()
endfunction()


