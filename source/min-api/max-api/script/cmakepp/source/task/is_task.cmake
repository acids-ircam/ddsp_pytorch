
function(is_task)
  map_get_special("${ARGN}" $type)
  ans(type)
  if("${type}" STREQUAL "task")
    return(true)
  endif()
  return(false)
endfunction()