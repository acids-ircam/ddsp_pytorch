

# returns the extension for a shell script file on the current console
# e.g. on windows this returns bat on unix/bash this returns bash
# uses shell_get() to determine which shell is used
function(shell_get_script_extension)
  shell_get()
  ans(shell)
  if("${shell}" STREQUAL "cmd")
    return(bat)
  elseif("${shell}" STREQUAL "bash")
    return(sh)
  else()
    message(FATAL_ERROR "no shell could be recognized")
  endif()

endfunction()