
  function(token_stream_new language str)
    map_get(${language}  token_definitions)
    ans(token_definitions)
   # messagE("new token strean ${token_definitions}")

    #address_print(${language})

    tokens_parse("${token_definitions}" "${str}")
    ans(tokens)
    map_new()
    ans(stream)
    map_set(${stream} current ${tokens})
    stack_new()
    ans(stack)
    map_set(${stream} stack ${stack})
    map_set(${stream} first ${tokens})
    return_ref(stream)
  endfunction()
