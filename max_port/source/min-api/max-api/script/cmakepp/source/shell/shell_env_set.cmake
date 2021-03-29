

# sets a system wide environment variable 
# the variable will not be available until a new console is started
function(shell_env_set key value)
  if(WIN32)
    reg_write_value("HKCU/Environment" "${key}" "${value}")
    #message("environment variable '${key}' was written, it will be available as soon as you restart your shell")
    return()
  endif()
  

  shell_get()
  ans(shell)
    
  if("${shell}" STREQUAL "bash")
    path("~/.bashrc")
    ans(path)
    fappend("${path}" "\nexport ${key}=${value}")
    #message("environment variable '${key}' was exported in .bashrc it will be available as soon as your restart your shell")
  else()
    message(WARNING "shell_set_env not implemented")
  endif()
endfunction()