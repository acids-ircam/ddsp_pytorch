## returns the <qualified directory> where the user data is stored
# this is the home dir/.cmakepp
function(user_data_dir)    
  home_dir()
  ans(home_dir)
  set(storage_dir "${home_dir}/.cmakepp")
  if(NOT EXISTS "${storage_dir}")
    mkdir("${storage_dir}")
  endif()
  return_ref(storage_dir)
endfunction()