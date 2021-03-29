# not finished
function(table_serialize)  
  objs(${ARGN})  
  ans(lines)


  map_new()
  ans(column_layout)

  set(allkeys)

  # get column_layout and col sizes
  foreach(line ${lines})
    map_keys(${line})
    ans(keys)
    
    foreach(key ${keys})  
      map_tryget(${column_layout} ${key})
      ans(res)
      
      map_tryget(${line} ${key})
      ans(val)
      string(LENGTH "${val}" len)
        
      if(${len} GREATER "0${res}")
        map_set(${column_layout} ${key} "${len}")
      endif()
    endforeach()
  endforeach()


  map_keys(${column_layout})
  ans(headers)
  set(res)
  set(separator)
  set(layout)
  set(first true)
  foreach(header ${headers})
    if(first)
      set(first false)
    else()
      set(res "${res} ")
      set(separator "${separator} ")
    endif()

    map_tryget(${column_layout} "${header}")
    ans(size)
    string_pad("${header}" "${size}")
    ans(header)    
    set(res "${res}${header}")
    string_repeat("=" "${size}")
    ans(sep)
    set(separator "${separator}${sep}")
  endforeach()

  set(res "${res}\n${separator}\n")
  

  foreach(line ${lines})
    set(first true)    
    foreach(header ${headers})
      if(first)
        set(first false)
      else()
        set(res "${res} ")      
      endif()
      map_tryget(${column_layout} "${header}")
      ans(size)
      map_tryget(${line} "${header}")
      ans(val)
      string_pad("${val}" ${size})
      ans(val)
      set(res "${res}${val}")
    endforeach()
    set(res "${res}\n")
  endforeach()

  return_ref(res)
endfunction()