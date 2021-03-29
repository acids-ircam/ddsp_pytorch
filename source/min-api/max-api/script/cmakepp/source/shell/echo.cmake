# writes the args to the console
function(echo)
  execute_process(COMMAND ${CMAKE_COMMAND} -E echo "${ARGN}")
endfunction()