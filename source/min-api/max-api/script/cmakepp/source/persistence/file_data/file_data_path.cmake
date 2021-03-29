


function(file_data_path dir id)
  path("${dir}/${id}.cmake")
  ans(path)
  return_ref(path)    
endfunction()