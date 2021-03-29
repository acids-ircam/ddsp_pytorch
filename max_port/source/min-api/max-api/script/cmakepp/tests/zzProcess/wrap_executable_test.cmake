function(test)

  fwrite("myscript.cmake" "
    set(asd 0)
     while(asd LESS 100) 
      message(STATUS waht) 
      math(EXPR asd \"\${asd}+1\")
     endwhile()
     ")
  ans(path)
  wrap_executable(cmake_script_test "${CMAKE_COMMAND}" -P "${path}")


  #cmake_script_test(--async-wait --process-handle)
  #ans(res)
  #assertf({res.pid} MATCHES "[1-9][0-9]*")


  
  assert(NOT COMMAND test_exectutable_wrapper)
  wrap_executable(test_exectutable_wrapper ${CMAKE_COMMAND})  

  ## assert that a function was created as specified
  assert(COMMAND test_exectutable_wrapper)


  fwrite("myscript.cmake" "message(STATUS hello)")
  fwrite("myerrorscript.cmake" "message(FATAL_ERROR byebye)")

  ## assert that --exit-code  returns the correct return code 
  ## and working directory is correct
  test_exectutable_wrapper(-P myscript.cmake --exit-code)
  ans(res)

  assert("${res}" EQUAL "0")

  ## assert that --exit-code returns a non 0 return code whne execution reports error
  test_exectutable_wrapper(-P myerrorscript.cmake --exit-code)
  ans(res)
  assert(NOT "${res}" EQUAL "0")


  ## assert that --result returns an object containing correct return code
  test_exectutable_wrapper(-P myscript.cmake --process-handle)
  ans(res)
  assert(DEREF "{res.exit_code}" STREQUAL "0") # return code should be error free
  assert(DEREF "{res.stdout}" MATCHES "hello") # stdout should contain only hello (possibly a line break)
  assert(DEREF "{res.start_info.timeout}" STREQUAL "-1") # timeout default value should be -1 (no timeout)

  test_exectutable_wrapper(-P myerrorscript.cmake --process-handle)
  ans(res)
  assert(DEREF NOT "{res.exit_code}" EQUAL "0")
  assert(DEREF "{res.stderr}" MATCHES "byebye") # stdout should contain only hello (possibly a line break)  
  assert(DEREF "{res.start_info.timeout}" STREQUAL "-1") # timeout default value should be -1 (no timeout)


  ## assert that no flag returns correct application output
  test_exectutable_wrapper(-P myscript.cmake)
  ans(res)
  assert("${res}" MATCHES "hello")


endfunction()