

    function(sequence_append map idx)
      sequence_count("${map}")
      ans(count)
      if(NOT "${idx}" LESS "${count}" OR ${idx} LESS 0)
        message(FATAL_ERROR "sequence_set: index out of range: ${idx}")
      endif()

      map_append( "${map}" "${idx}" ${ARGN} )
      
    endfunction()