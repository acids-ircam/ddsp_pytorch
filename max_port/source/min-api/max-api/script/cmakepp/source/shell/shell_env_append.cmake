

function(shell_env_append key value)
  if(WIN32)
    shell("SETX ${key} %${key}%;${value}")

  else()
    message(WARNING "shell_set_env not implemented for anything else than windows")

  endif()
endfunction()