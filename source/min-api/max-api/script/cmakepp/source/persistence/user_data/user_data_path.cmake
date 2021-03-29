## returns the user data path for the specified id
## id can be any string that is also usable as a valid filename
## it is located in %HOME_DIR%/.cmakepp
function(user_data_path id)  
  if(NOT id)
    message(FATAL_ERROR "no id specified")
  endif()
  user_data_dir()
  ans(storage_dir)
  set(storage_file "${storage_dir}/${id}.cmake")
  return_ref(storage_file)
endfunction()

