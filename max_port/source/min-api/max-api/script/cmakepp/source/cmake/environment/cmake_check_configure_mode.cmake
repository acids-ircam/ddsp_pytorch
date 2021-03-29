## causes an error if cmake is not in configure mode
function(cmake_check_configure_mode)
  cmake_is_configure_mode()
  ans(isConfigMode)
  if(NOT isConfigMode)
    message(FATAL_ERROR "cmake needs to be in config mode! - aborting execution")
  endif()
  return()
endfunction()
