# writes the args to console. does not append newline
function(echo_append)
  execute_process(COMMAND ${CMAKE_COMMAND} -E echo_append "${ARGN}")
endfunction()