function(ast_parse_sequence )#definition stream create_node definition_id
  map_tryget("${definition}"  sequence)
  ans(sequence)
  set(rsequence)
  if(NOT sequence)
    map_tryget("${definition}"  rsequence)
    ans(sequence)
    set(rsequence true)
  endif()
  if(NOT sequence)
    message(FATAL_ERROR "expected a sequence or a rsequence")
  endif()
  # deref ref array
#  address_get(${sequence} )
#  ans(sequence)
  
  # save current stream
  #message("push")
  token_stream_push(${stream})

  # empty var for sequence
  set(ast_sequence)

  # loop through all definitions in sequence
  # adding all resulting nodes in order to ast_sequence
  foreach(def ${sequence})
    ast_parse(${stream} "${def}")
    ans(res)
    if(res) 
      is_map(${res} )
      ans(ismap)
      if(ismap)
        list(APPEND ast_sequence ${res})
      endif()
    else()
     # message("pop")
      token_stream_pop(${stream})
      return(false)
    endif()
   
  endforeach()
  token_stream_commit(${stream})
  # return result
  if(NOT create_node)
    return(true)
  endif()
  map_new()
  ans(node)
  map_set(${node} types ${definition_id})
  
  map_set(${node} children ${ast_sequence})
  return(${node})
endfunction()
