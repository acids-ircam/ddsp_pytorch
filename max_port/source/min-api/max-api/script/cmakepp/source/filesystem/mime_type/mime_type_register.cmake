

## https://www.ietf.org/rfc/rfc2045.txt
function(mime_type_register mime_type)
  data("${mime_type}")
  ans(mime_type)

  map_tryget("${mime_type}" name)
  ans(name)
  if(name STREQUAL "")
    return()
  endif()

  mime_type_map()
  ans(mime_types)

  map_tryget("${mime_types}" "${name}")
  ans(existing_mime_type)
  if(existing_mime_type)
    message(FATAL_ERROR "mime_type ${name} already exists")
  endif()

  map_tryget("${mime_type}" extensions)
  ans(extensions)


  foreach(key ${name} ${extensions})
    map_append(${mime_types} "${key}" "${mime_type}")
  endforeach()

  return_ref(mime_type)

endfunction()

