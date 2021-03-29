

function(tokenize)
  arguments_tokenize("0" "${ARGC}")
  set(type_list ${type_list} PARENT_SCOPE)
  set(error ${error} PARENT_SCOPE)
  set(token_types ${token_types} PARENT_SCOPE)
  set(tokens ${tokens} PARENT_SCOPE)
endfunction()
