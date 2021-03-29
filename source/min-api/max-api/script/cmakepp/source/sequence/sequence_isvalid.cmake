
    function(sequence_isvalid map)
      sequence_count("${map}")
      ans(is_lookup)

      if("${is_lookup}_" STREQUAL "_" )
        return(false)
      endif()
      return(true)
    endfunction()