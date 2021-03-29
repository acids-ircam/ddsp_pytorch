
 function(print_multi n)


  set(headers index ${ARGN})
  set(header_lengths )
  foreach(header ${headers})
    string(LENGTH "${header}" header_len)
    math(EXPR header_len "${header_len} + 1")
    list(APPEND header_lengths ${header_len})
  endforeach()

  string(REPLACE ";" " " headers "${headers}")
  message("${headers}")

  if(${n} LESS 0)
    return()
  endif()

  foreach(i RANGE 0 ${n})
    set(current_lengths ${header_lengths})
    list_pop_front(current_lengths )
    ans(current_length)
    echo_append_padded("${current_length}" "${i}")
    foreach(arg ${ARGN})

      list_pop_front(current_lengths )
      ans(current_length)
      is_map("${${arg}}")
      ans(ismap)
      if(ismap)
        map_tryget(${${arg}} ${i})
        ans(val)
      else()
        list(GET ${arg} ${i} val)
      endif()

      echo_append_padded("${current_length}" "${val}")
    endforeach()  
    message(" ")
  endforeach()
 endfunction()