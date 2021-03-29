
function(file_isqmfile file)
    path_qualify(file)
    if(NOT EXISTS "${file}" OR IS_DIRECTORY "${file}")
      return(false)
    endif()
  file(READ "${file}" result LIMIT 3)
  if(result STREQUAL "#qm")
    return(true)
  endif()

  return(false)

endfunction()