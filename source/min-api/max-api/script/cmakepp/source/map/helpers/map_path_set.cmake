# todo implement

function(map_path_set map path value)
  message(FATAL_ERROR "not implemented")
  if(NOT map)
    map_new()
    ans(map)
  endif()

  set(current "${map}")

  foreach(arg ${path})
    map_tryget("${current}" "${arg}")
    ans(current) 

  endforeach()

endfunction()
