function(test)
  
  if(NOT WIN32)
    message("Test Inconclusive: Powershell is only available on windows >= XP")
    return()
  endif()



  win32_powershell_run_script("echo hello")
  ans(res)
  assert("${res}" MATCHES "hello")

endfunction()