function(ast_parse_any )#definition stream create_node definition_id
  # check if definition contains "any" property
  map_tryget(${definition}  any)
  ans(any)
#  address_get(${any})
#  ans(any)
  
  # try to parse any of the definitions contained in "any" property
  set(node false)
  foreach(def ${any})    
    ast_parse(${stream} "${def}")
    ans(node)
    if(node)
      break()
    endif()
  endforeach()

  # append definition to current node if a node was returned
  is_address("${node}")
  ans(is_map)
  if(is_map)
  
    map_append(${node} types ${definition_id})
  endif()
  
  
  
  return_ref(node)
endfunction()
