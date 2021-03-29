
function(cmake_is_configure_mode)
  cmake_is_script_mode()
  ans(res)
  if(res)
    return(false)
  else()
    return(true)
  endif()
endfunction()

