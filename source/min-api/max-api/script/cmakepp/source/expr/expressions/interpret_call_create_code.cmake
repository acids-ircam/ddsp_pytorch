
function(interpret_call_create_code output_ref callable_ast parameter_asts)

  set(parameters_string)
  ## insert parameters passed as argn
  foreach(parameter ${parameter_asts})

    map_tryget("${parameter}" expression_type)
    ans(parameter_expression_type)
    #print_Vars(parameter.expression_type parameter.value)
    if("${parameter_expression_type}" STREQUAL "ellipsis")
      map_tryget("${parameter}" children)
      ans(parameter)
      map_tryget("${parameter}" value)
      ans(parameter_value)
      set(parameters_string "${parameters_string} ${parameter_value}")
    else()
      map_tryget("${parameter}" value)
      ans(parameter_value)
      set(parameters_string "${parameters_string} \"${parameter_value}\"")
    endif()   
  endforeach()

  ## remove initial space
  if(parameter_asts)
    string(SUBSTRING "${parameters_string}" 1 -1 parameters_string)
  endif()


  map_tryget("${callable_ast}" const)
  ans(callable_is_const)

  map_tryget("${callable_ast}" value)
  ans(callable_value)

  if(callable_is_const)
    set(code "${callable_value}(${parameters_string})\nans(${output_ref})\n")
  else()

function(cmake_string_escape3 str)
  if("${str}" MATCHES "[ \"\\(\\)#\\^\t\r\n\\]")
    ## encoded list encode cmake string...
    #string(REPLACE "\\" "\\\\" str "${str}")
    string(REGEX REPLACE "([ \"\\(\\)#\\^])" "\\\\\\1" str "${str}")
    string(REPLACE "\t" "\\t" str "${str}")
    string(REPLACE "\n" "\\n" str "${str}")
    string(REPLACE "\r" "\\r" str "${str}")  
  endif()
  return_ref(str)
endfunction()

    cmake_string_escape3("${parameters_string}")
    ans(parameters_string)
    set(code "eval(\"${callable_value}(${parameters_string})\")\nans(${output_ref})\n")
  endif()


  ## set this if a this value is present
  map_tryget("${callable_ast}" this)
  ans(this)
  if(this)
    map_tryget("${this}" value)
    ans(this_value)
    set(code "set(this ${this_value})\n${code}")
  endif()



  return_ref(code)

endfunction()