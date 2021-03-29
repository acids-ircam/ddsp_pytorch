
## causes an error if cmake is not in script mode
function(cmake_check_script_mode)
  cmake_is_script_mode()
  ans(isScriptMode)
  if(NOT isScriptMode)
    message(FATAL_ERROR "cmake needs to be in script mode! - aborting execution")
  endif()
  return()
endfunction()

