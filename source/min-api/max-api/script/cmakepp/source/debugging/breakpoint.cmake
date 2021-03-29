# creates a breakpoint 
# usage: breakpoint(${CMAKE_CURRENT_LIST_FILE} ${CMAKE_CURRENT_LIST_LINE})
function(breakpoint file line) 
  if(NOT DEBUG_CMAKE)
    return()
  endif()
  message("breakpoint reached ${file}:${line}")
  while(1)
    echo_append("> ")
    read_line()
    ans(cmd)
    if("${cmd}" STREQUAL "")
      message("continuing execution")
      break()
    endif()

    
    if("${cmd}" MATCHES "^\\$.*")
      string(SUBSTRING "${cmd}" 1 -1 var)
      

      get_cmake_property(_variableNames VARIABLES)
      foreach(v ${_variableNames})
        if("${v}" MATCHES "${cmd}")
          dbg("${v}")

        endif()
      endforeach()

    endif()
    



  endwhile()
endfunction()
