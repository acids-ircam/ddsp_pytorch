function(test)

  ## the workind directory needs to exist. maybe this needs to be ensured within execute.
  mkdir(mytempdir)
  
  ## check if command is actually started in working directory
  ## in sync mode
  execute_script("echo_append(\${CMAKE_CURRENT_BINARY_DIR})" WORKING_DIRECTORY "mytempdir") # delegates to cmake
  ans(res)
  assert("${res}" MATCHES "/mytempdir$")  


  ## check if command is actually started in working directory 
  ## in async mode
  execute_script("echo_append(\${CMAKE_CURRENT_BINARY_DIR})" WORKING_DIRECTORY "mytempdir" --async)
  ans(process_handle)

  process_wait(${process_handle})
  assertf("{process_handle.stdout}" MATCHES "/mytempdir$")


##
  mkdir("dir1")
  fwrite(dir1/successscript.cmake "message(STATUS hello)")
  fwrite(dir1/errorscript.cmake "message(FATAL_ERROR byebye)")

  ## act
  execute(${CMAKE_COMMAND} -P successscript.cmake TIMEOUT 3 WORKING_DIRECTORY dir1 --process-handle)
  ans(res)

  ## assert
  assertf("{res.start_info.command}" STREQUAL "${CMAKE_COMMAND}")
  assertf("{res.start_info.command_arguments}" CONTAINS -P  )
  assertf("{res.start_info.command_arguments}" CONTAINS successscript.cmake  )

  path("dir1")
  ans(cwd)
  assertf("{res.start_info.working_directory}" STREQUAL "${cwd}")
  assertf("{res.exit_code}" EQUAL 0)

  assertf("{res.stdout}" MATCHES "hello")
  assertf("{res.start_info.timeout}" EQUAL 3)

  ## act
  execute(${CMAKE_COMMAND} -P errorscript.cmake WORKING_DIRECTORY dir1 --process-handle)
  ans(res)

  ## assert
  assertf("{res.start_info.command}" STREQUAL "${CMAKE_COMMAND}")
  assertf("{res.start_info.command_arguments}" CONTAINS -P )
  assertf("{res.start_info.command_arguments}" CONTAINS errorscript.cmake )

  path("dir1")
  ans(cwd)
  assertf("{res.start_info.working_directory}" STREQUAL "${cwd}")
  assertf("{res.stderr}" MATCHES "byebye")
  assertf(NOT "{res.exit_code}" EQUAL 0)



  execute(COMMAND cmake -E echo_append "hello;hello" turn the radio on --process-handle)
  ans(res)
  assertf("{res.stdout}" EQUALS "hello;hello turn the radio on" )

  map_new()
  ans(context)
  execute(COMMAND cmake -E echo_append "hello;hello" 
    --success-callback "[](handle)map_append(${context} result 'success{{exit_code}}')" 
    --error-callback "[](handle) map_append(${context} result 'error{{exit_code}}')"
    --silent-fail
  )

  assertf("{context.result}" STREQUAL "success0")



  map_new()
  ans(context)
  execute(COMMAND cmake -E echo_appendFAILBECAUSESTUPIDERROR "hello;hello" 
    --success-callback "[](handle)map_append(${context} result 'success{{exit_code}}')" 
    --error-callback "[](handle) map_append(${context} result 'error{{exit_code}}')"
    --silent-fail
  )

  assertf("{context.result}" STREQUAL "error1")

endfunction()