
  function(ast_parse_token )#definition stream create_node definition_id
    #message(FORMAT "trying to parse {definition.name}")
   # address_print("${definition}")
   # address_print(${definition})

    token_stream_take(${stream} ${definition})
    ans(token)

    if(NOT token)
      return(false)
    endif()
    
    #message(FORMAT "parsed {definition.name}: {token.data}")
    if(NOT create_node)
      return(true)
    endif()

    map_tryget(${definition}  replace)
    ans(replace)
    if(replace)
      map_get(${token}  data)
      ans(data)
      map_get(${definition}  regex)
      ans(regex)
      string(REGEX REPLACE "${regex}" "\\${replace}" data "${data}")
      #message("data after replace ${data}")
      map_set_hidden(${token} data "${data}")
    endif()
    
    map_set_hidden(${token} types ${definition_id})
    return(${token})

  endfunction()