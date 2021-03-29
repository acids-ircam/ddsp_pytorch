

  function(map_conditional_evaluate parameters)
    set(result)
    foreach(map ${ARGN})
      map_conditional_single("${parameters}" "${map}")
      ans_append(result)
    endforeach()
    return_ref(result)
  endfunction()