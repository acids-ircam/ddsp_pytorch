
  function(propref_get_ref)
    string_split_at_last(ref prop "${propref}" ".")
    return_ref(ref)
  endfunction()