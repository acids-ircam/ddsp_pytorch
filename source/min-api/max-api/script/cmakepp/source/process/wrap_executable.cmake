
# wrap_executable(<alias> <executable> <args...>)-><null>
#
# creates a function called ${alias} which wraps the executable specified in ${executable}
# <args...> will be set as command line arguments for every call
# the alias function's varargs will be passed on as command line arguments. 
#
# Warning: --async is a bit experimental
#
# defines function
# <alias>([--async]|[--process-handle]|[--exit-code])-> <stdout>|<process result>|<exit code>|<process handle>
#
# <no flag>       if no flag is specified then the function will fail if the return code is not 0
#                 if it succeeds the return value is the stdout
#
# --process-handle        flag the function will return a the execution 
#                 result object (see execute()) 
# --exit-code     flag the function will return the exit code
# --async         will execute the executable asynchroniously and
#                 return a <process handle>
# --async-wait    will execute the executable asynchroniously 
#                 but will not return until the task is finished
#                 printing a status indicator
# --lean          lean call to executable (little overhead - no events etc)
# 
# else only the application output will be returned 
# and if the application terminates with an exit code != 0 a fatal error will be raised
function(wrap_executable alias executable)
  arguments_encoded_list(0 ${ARGC})
  ans(arguments)
  # remove alias and executable
  list_pop_front(arguments)
  list_pop_front(arguments)

  eval("  
    function(${alias})
      arguments_encoded_list(0 \${ARGC})
      ans(arguments)
      execute(\"${executable}\" ${arguments} \${arguments})
      return_ans()
    endfunction()
    ")
  return()
endfunction()
