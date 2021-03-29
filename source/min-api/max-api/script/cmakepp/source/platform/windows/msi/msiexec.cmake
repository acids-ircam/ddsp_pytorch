


# wraps the win32 console executable cmd.exe
function(msiexec)
  if(NOT WIN32)
    message(FATAL_ERROR "not supported on your os - only Windows")
  endif()
  wrap_executable(msiexec msiexec.exe)
  msiexec(${ARGN})
  return_ans()
endfunction()
