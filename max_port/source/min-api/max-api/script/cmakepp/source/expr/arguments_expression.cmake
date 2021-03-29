## `(<begin index> <end index> <parameter definition>...)-><any>...`
##
##
## parses the arguments of the calling function cmakepp expressions
## expects `begin index` to be the index of first function parameters (commonly 0)
## and `end index` to be the index of the last function parameter to parse (commonly ${ARGC})
## var args are named arguments which will be set to be available in the function scope
##
## named arguments passed to function have a higher precedence than positional arguments 
##
## __sideffects__:
## * `arguments_expression_result` is a address of an object containing all parsed data
## * scope operations may modify the parent scope of the function
## 
##
macro(arguments_expression begin end)
  ## eval arguments expression
  ## see interpret_cmake_function_parameters 
  set(_argn_ ${ARGV})
  arguments_expression_eval_cached(
   interpret_cmake_function_parameters
   "" 
   "_argn_" 
   ${begin} 
   ${end}
  )
  ans(arguments_expression_result)

  ## set all positional values
  address_get(${arguments_expression_result})
  ans(__positionals)
  foreach(arg ${ARGN})
    map_has("${arguments_expression_result}" "${arg}")
    ans(has_named_value)
    if(has_named_value)
      map_tryget("${arguments_expression_result}" "${arg}")
      ans(${arg})
    else()
      list_pop_front(__positionals)
      encoded_list_decode("${__ans}")
      ans(${arg})  
    endif()

    
  endforeach()
  ## return the rest of the positional values which were not assign
  set(__ans ${__positionals})

endmacro()