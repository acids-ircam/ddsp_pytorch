## a fast wrapper for the specified executable
## this should be used for executables that are called often
## and do not need to run async
function(wrap_executable_bare alias executable)

  eval("
    function(${alias})
      set(args \${ARGN})
      list_extract_flag(args --passthru)
      ans(passthru)
      pwd()
      ans(cwd)      
      set(output)
      set(stdout)
      if(NOT passthru)
        set(output 
          OUTPUT_VARIABLE stdout 
          ERROR_VARIABLE stdout 
          )
      endif()
      execute_process(COMMAND \"${executable}\" ${ARGN} \${args}
        WORKING_DIRECTORY  \"\${cwd}\"
        \${output}
        RESULT_VARIABLE error
      )
      list(INSERT stdout 0 \${error})
      return_ref(stdout)
    endfunction()
    ")
  return()
endfunction()