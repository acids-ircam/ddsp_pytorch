

    function(sequence_add map)
      sequence_count("${map}")
      ans(count)
      math(EXPR new_count "${count} + 1")
      map_set_special("${map}" count ${new_count})
      map_set("${map}" "${count}" ${ARGN})
      return_ref(count)
    endfunction()
