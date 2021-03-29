
## returns structured data - either from a runtime object or a file
function(structured_data)
  is_map("${ARGN}")
  ans(isMap)
  if(isMap)
    return(${ARGN})
  endif()

  fopen_data("${ARGN}")
  ans(data)

  return_ref(data)
endfunction()