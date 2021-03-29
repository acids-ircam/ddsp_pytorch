
# parses a structured list given the structure map
# returning a map which contains all the parsed values
function(structured_list_parse structure_map)
  map_new()
  ans(result)
  set(args ${ARGN})
  obj("${structure_map}")
  ans(structure_map)

  if(NOT structure_map)
    return_ref(result)
  endif() 

  # get all keys
  map_keys(${structure_map} )
  ans(keys)
  set(cutoffs)

  # parse every value descriptor from structure map
  # add every label to the list of cutoffs (a new element definition cuts othe rvalues)
  set(descriptors)
  foreach(key ${keys})
    map_tryget(${structure_map}  "${key}")
    ans(current)
    if(current)
      value_descriptor_parse(${key} ${current})
      ans(current_descriptor)

      list(APPEND descriptors ${current_descriptor})
      map_tryget(${current_descriptor}  "labels")
      ans(labels)
      list(APPEND cutoffs ${labels})        
    endif()
  endforeach()

  # go through each descriptor
  set(errors)
  foreach(current_descriptor ${descriptors})
    nav(labels = current_descriptor.labels)
    nav(id = current_descriptor.id)
    list(REMOVE_ITEM cutoffs ${labels})

    set(error)
    list_parse_descriptor(${current_descriptor} ERROR error UNUSED_ARGS args CUTOFFS cutoffs ${args} )
    #message(FORMAT "args left ${args} after {current_descriptor.id}")
    ans(current_result)
    if(NOT current_result)
      nav(current_result = current_descriptor.default)
    endif()
    if(error)
      list(APPEND errors ${id})
    endif()
    string_decode_semicolon("${current_result}")
    ans(current_result)
    map_navigate_set("result.${id}" ${current_result})
  endforeach()
  #message("args left ${args}")
  map_navigate_set("result.unused" "${args}")
  map_navigate_set("result.errors" "${errors}")
  #message("errors ${errors}")
  return(${result})
endfunction()