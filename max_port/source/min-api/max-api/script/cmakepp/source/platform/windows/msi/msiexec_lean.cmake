
#
function(msiexec_lean)
    if(NOT WIN32)
    message(FATAL_ERROR "not supported on your os - only Windows")
  endif()

  wrap_executable_bare(msiexec_lean msiexec.exe)
  msiexec_lean(${ARGN})
  return_ans()
endfunction()