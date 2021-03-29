## returns the <return_code> for the specified process handle
## if process is not finished the result is empty
  function(process_return_code handle)
    process_handle("${handle}")
    ans(handle)
    map_tryget("${handle}" return_code_file)
    ans(return_code_file)
    fread("${return_code_file}")
    ans(return_code)
    string(STRIP "${return_code}" return_code)
    return_ref(return_code)
  endfunction()
