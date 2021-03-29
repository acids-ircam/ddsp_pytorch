### returns the user data stored under the index id
## user data may be any kind of data  
function(user_data_read id)
  user_data_path("${id}")
  ans(storage_file)

  if(NOT EXISTS "${storage_file}")
    return()
  endif()

  qm_read("${storage_file}")
  return_ans()
endfunction()