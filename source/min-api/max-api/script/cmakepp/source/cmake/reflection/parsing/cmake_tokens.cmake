## `(<cmake code>|<cmake token>...)-><cmake token>...`
##
## coerces the input to a token list
function(cmake_tokens tokens)
  string_codes()

  if("${tokens}" MATCHES "^${ref_token}:")
    return_ref(tokens)
  endif()
  cmake_tokens_parse("${tokens}" --extended)
  rethrow()
  return_ans()
endfunction()