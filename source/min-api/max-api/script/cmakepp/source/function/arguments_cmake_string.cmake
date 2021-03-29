##
##
## recreates the invocation arguments which when evaluated are identical
## to the arguments passed to the function where this macro was invoked
macro(arguments_cmake_string __arg_begin __arg_end)
  set(__arg_res)   
  if(${__arg_end} GREATER ${__arg_begin})
    math(EXPR __last_arg_index "${__arg_end} - 1")
    foreach(i RANGE ${__arg_begin} ${__last_arg_index} )        
      cmake_string_escape("${ARGV${i}}")
      set(__arg_res "${__arg_res} \"${__ans}\"")
    endforeach()
    string(SUBSTRING "${__arg_res}" 1 -1 __ans)
  else()
    set(__ans)
  endif()
endmacro()