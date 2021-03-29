
  function(encoded_list_append __lst)
    string_encode_list("${ARGN}")
    list(APPEND "${__lst}" ${__ans})
    set(${__lst} ${${__lst}} PARENT_SCOPE)
  endfunction()
