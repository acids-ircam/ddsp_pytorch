
  function(map_flatten)
    set(result)
    foreach(map ${ARGN})
      map_values(${map})
      ans_append(result)
    endforeach()
    return_ref(result)
  endfunction()