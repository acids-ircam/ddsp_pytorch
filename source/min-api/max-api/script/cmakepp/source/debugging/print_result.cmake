
#prints result
function(print_result result)
  list(LENGTH argc "${result}" )
  if("${argc}" LESS 2)
    message("${result}")
  else()
    foreach(arg ${result})
      message("${arg}")
    endforeach()
  endif()
endfunction()
