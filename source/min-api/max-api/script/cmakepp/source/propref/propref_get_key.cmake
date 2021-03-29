
  function(propref_get_key)
    string_split_at_last(ref prop "${propref}" ".")
    return_ref(prop)
  endfunction()
  
 