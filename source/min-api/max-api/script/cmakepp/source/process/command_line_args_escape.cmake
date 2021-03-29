## escapes a command line quoting arguments as needed 
function(command_line_args_escape) 
  set(whitespace_regex "( )")
  set(result)
  
  string(ASCII  31 us)

  foreach(arg ${ARGN})
    string(REGEX MATCH "[\r\n]" m "${arg}")
    if(NOT "_${m}" STREQUAL "_")
      message(FATAL_ERROR "command line argument '${arg}' is invalid - contains CR NL - consider escaping")
    endif()

    string(REGEX MATCH "${whitespace_regex}|\"" m "${arg}")
    if("${arg}" MATCHES "${whitespace_regex}|\"")
      string(REPLACE "\"" "\\\"" arg "${arg}")
      set(arg "\"${arg}\"")
    elseif("${arg}" MATCHES "${us}")
      set(arg "\"${arg}\"")
    endif()




    list(APPEND result "${arg}")

  endforeach()    
  return_ref(result)
endfunction()
