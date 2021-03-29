function(test_execute test)
  event_addhandler(on_log_message "[](msg) message(FORMAT '{msg.function}> {msg.message}') ")
  ans(handler)

  message(STATUS "running test ${test}...")

  #initialize variables which test can use

  get_filename_component(test_name "${test}" NAME_WE) 

  # intialize message listener

  # setup a directory for the test
  string_normalize("${test_name}")
  ans(test_dir)
  cmakepp_config(temp_dir)
  ans(temp_dir)
  set(test_dir "${temp_dir}/tests/${test_dir}")
  file(REMOVE_RECURSE "${test_dir}")
  get_filename_component(test_dir "${test_dir}" REALPATH)
  path_qualify(test)
  message(STATUS "test directory is ${test_dir}")  
  pushd("${test_dir}" --create)
  timer_start("test duration")


  call("${test}"())
  
  set(time)
  timer_elapsed("test duration")
  ans(time)
  popd()

  event_removehandler(on_log_message ${handler})


  message(STATUS "done after ${time} ms")
endfunction()