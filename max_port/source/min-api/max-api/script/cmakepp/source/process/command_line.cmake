
function(command_line)
  is_map("${ARGN}")
  ans(ismap)
  if(ismap)
    map_has("${ARGN}" command)
    ans(iscommand_line)
    if(iscommand_line)
      return("${ARGN}")
    endif()
    return()
  endif()
  command_line_parse(${ARGN})
  return_ans()


endfunction()