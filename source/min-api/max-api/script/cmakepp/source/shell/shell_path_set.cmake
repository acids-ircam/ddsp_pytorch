

function(shell_path_set)
  set(args ${ARGN})
  if(WIN32)
    string(REPLACE "\\\\" "\\" args "${args}")
  endif()
  message("setting path ${args}")
  shell_env_set(Path "${args}")
  return()
endfunction()