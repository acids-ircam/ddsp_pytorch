

function(file_data_read dir id)
  file_data_path("${dir}" "${id}")      
  ans(path)
  if(NOT EXISTS "${path}")
    return()
  endif()
  qm_read("${path}")
  return_ans()
endfunction()
