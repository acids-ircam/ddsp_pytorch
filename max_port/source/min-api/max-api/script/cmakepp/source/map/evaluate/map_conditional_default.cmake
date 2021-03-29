

  function(map_conditional_default parameters) 
    set(value ${ARGN})   
    map_clone_shallow("${value}")
    ans(value)
    
    is_map("${value}")
    ans(is_map)
    if(NOT is_map)
      return_ref(value)
    endif()


    foreach(key ${keys})
      map_tryget("${value}" "${key}")
      ans(val)
      map_conditional_evaluate("${parameters}" ${val})
      ans(val)
      map_set("${value}" "${key}" "${val}")
    endforeach()



    return(${value})

  endfunction()