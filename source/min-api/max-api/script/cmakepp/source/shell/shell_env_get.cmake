
# returns the value of the shell's environment variable ${key}
function(shell_env_get key)
  shell_get()
  ans(shell)

  if(WIN32)
    
  endif()

  if("${shell}" STREQUAL "cmd")
    #setlocal EnableDelayedExpansion\nset val=\nset /p val=\necho %val%> \"${value_file}\"
    shell_redirect("echo %${key}%")
    ans(res)
  elseif("${shell}" STREQUAL "bash")
    shell_redirect("echo $${key}")
    ans(res)
  else()
    message(FATAL_ERROR "${shell} not supported")
  endif()


    # strip trailing '\n' which might get added by the shell script. as there is no way to input \n at the end 
    # manually this does not change for any system
    if("${res}" MATCHES "(\n|\r\n)+$")
      string(REGEX REPLACE "(\n|\r\n)+$" "" res "${res}")
    endif()
    
  return_ref(res)
endfunction()