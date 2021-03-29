# returns the current working directory
  function(pwd)
    address_get(__global_cd_current_directory)
    return_ans()
  endfunction()
