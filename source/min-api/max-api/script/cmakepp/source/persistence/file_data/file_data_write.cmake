


function(file_data_write dir id)
  file_data_path("${dir}" "${id}")
  ans(path)
  qm_write("${path}" ${ARGN})
  return_ref(path)
endfunction()