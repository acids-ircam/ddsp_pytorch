

    function(sequence_append_string map idx)
      sequence_count("${map}")
      ans(count)
      if(NOT "${idx}" LESS "${count}" OR ${idx} LESS 0)
        message(FATAL_ERROR "sequence_set: index out of range: ${idx}")
      endif()

      map_append_string( "${map}" "${idx}" ${ARGN} )
      
    endfunction()