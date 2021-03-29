
  function(encoded_list_set __lst idx)
    string_encode_list("${ARGN}")
    list_replace_at(${__lst} ${idx} ${__ans})
    set(${__lst} ${${__lst}} PARENT_SCOPE)
  endfunction()
