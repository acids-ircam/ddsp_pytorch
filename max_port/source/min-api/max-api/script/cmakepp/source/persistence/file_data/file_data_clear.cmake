

function(file_data_clear dir id)
  file_data_path("${dir}" "${id}")
  ans(path)
  if(NOT EXISTS "${path}")
    return(false)
  endif()
  rm("${path}")
  return(true)
endfunction()