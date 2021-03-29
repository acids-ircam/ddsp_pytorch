

    function(sequence_new)
      is_address("${ARGN}")
      ans(isref)
      if(NOT isref)
        map_new()
        ans(map)
      else()
        set(map ${ARGN})
      endif()

      map_set_special(${map} count 0)
      return_ref(map)
    endfunction()