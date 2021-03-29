## returns the argument string which was passed to the parent function
## it takes into considerations quoted arguments
## todo: start and endindex
macro(arguments_string __arg_begin __arg_end)
  set(__arg_res)   
  if(${__arg_end} GREATER 0)
    math(EXPR __last_arg_index "${__arg_end} - 1")
    foreach(i RANGE 0 ${__last_arg_index})
      set(__current "${ARGV${i}}")
      if("${__current}_" MATCHES "(^_$)|(;)|(\\\")")
        set(__current "\"${__current}\"")
      endif()
      set(__arg_res "${__arg_res} ${__current}")
    endforeach()
    string(SUBSTRING "${__arg_res}" "1" "-1" __ans)  
  else()
    set(__ans)
  endif()
endmacro()
