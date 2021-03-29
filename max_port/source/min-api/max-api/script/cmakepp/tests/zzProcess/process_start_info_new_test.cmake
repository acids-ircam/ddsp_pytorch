function(test)



  timer_start(t1)
  foreach(i RANGE 0 100)
    process_start_info_new(COMMAND cmake -E echo "a;b" "c;d")
  endforeach()
  timer_print_elapsed(t1)

  process_start_info_new(COMMAND cmake -E echo "a;b" "c;d")
  ans(res)
  assert(res)
  assertf({res.command} STREQUAL "cmake")
  assertf({res.command_arguments} CONTAINS "-E")
  assertf({res.command_arguments} CONTAINS "echo")
  assertf({res.working_directory} STREQUAL "${test_dir}")
  assertf({res.timeout} STREQUAL "-1")

  process_start_info_new(cmake -E echo "a;b" "c;d")
  ans(res)
  assert(res)
  assertf({res.command} STREQUAL "cmake")
  assertf({res.command_arguments} CONTAINS "-E")
  assertf({res.command_arguments} CONTAINS "echo")
  assertf({res.working_directory} STREQUAL "${test_dir}")
  assertf({res.timeout} STREQUAL "-1")


  process_start_info_new(cmake -E echo "a;b" "c;d")
  ans(res)
  assert(res)
  assertf({res.command} STREQUAL "cmake")
  assertf({res.command_arguments} CONTAINS "-E")
  assertf({res.command_arguments} CONTAINS "echo")
  assertf({res.working_directory} STREQUAL "${test_dir}")
  assertf({res.timeout} STREQUAL "-1")


  process_start_info_new(cmake -E echo "a;b" "c;d" TIMEOUT 2)
  ans(res)
  assert(res)
  assertf({res.command} STREQUAL "cmake")
  assertf({res.command_arguments} CONTAINS "-E")
  assertf({res.command_arguments} CONTAINS "echo")
  assertf({res.working_directory} STREQUAL "${test_dir}")
  assertf({res.timeout} STREQUAL "2")
  assertf({res.passthru} STREQUAL "false")



  process_start_info_new(cmake -E echo "a;b" "c;d" TIMEOUT 2 --passthru)  
  ans(res)
  assertf({res.passthru} STREQUAL "true")
return()
endfunction()