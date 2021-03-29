
  function(ast_parse_match definition stream create_node)
    # check if definition can be parsed by ast_parse_match
    map_tryget("${definition}"  match)
    ans(match)
    if(NOT match)
      return(false)
    endif()

    # take string specified in match from stream (if stream does)
    # not start with "${match}" nothing is returned
   # message("matching match ${match}")
#    stream_print(${stream})
    stream_take_string(${stream} "${match}")
    ans(res)
    # could not parse if stream did not match "${match}"
    if("${res}_" STREQUAL "_")
      return(false)
    endif()

    # return result
    if(NOT create_node)
      return(true)
    endif()
    map_new(node)
    ans(node)
    map_set(${node} data ${data})
    return(${node})
 endfunction()