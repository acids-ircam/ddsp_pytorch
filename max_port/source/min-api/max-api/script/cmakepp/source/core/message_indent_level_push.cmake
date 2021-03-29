

function(message_indent_push)
  
  set(new_level ${ARGN})
  if("${new_level}_" STREQUAL "_")
    set(new_level +1)
  endif()
  
  if("${new_level}" MATCHES "[+\\-]")
    message_indent_level()
    ans(previous_level)
    math(EXPR new_level "${previous_level} ${new_level}")
    if(new_level LESS 0)
      set(new_level 0)
    endif()
  endif()
  map_push_back(global message_indent_level ${new_level})
  return(${new_level})
endfunction()