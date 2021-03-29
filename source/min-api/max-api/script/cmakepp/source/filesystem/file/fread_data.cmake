

## tries to read the spcified file format
function(fread_data path)
  set(args ${ARGN})

  path_qualify(path)
  
  list_pop_front(args)
  ans(mime_type)

  if(NOT mime_type)

    mime_type("${path}")
    ans(mime_type)

    if(NOT mime_type)
      return()
    endif()

  endif()
  set(result)
  if("${mime_type}" MATCHES "application/json")
    json_read("${path}")
    ans(result)
  elseif("${mime_type}" MATCHES "application/x-quickmap")
    qm_read("${path}")
    ans(result)
  elseif("${mime_type}" MATCHES "application/x-serializedcmake")
    cmake_read("${path}")
    ans(result)
  else()
    return()
  endif()

  ## set target file property which allows identification of where the map was read
  ## if it was a single map
  map_source_file_set("${result}" "${path}")      

  return(${result})

endfunction()
