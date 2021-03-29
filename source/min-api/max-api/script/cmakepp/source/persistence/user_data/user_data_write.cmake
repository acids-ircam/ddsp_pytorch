## writes all var args into user data, accepts any typ of data 
## maps are serialized
function(user_data_write id)
  user_data_path("${id}")
  ans(path)
  qm_write("${path}" ${ARGN})
  return_ans()
endfunction()
