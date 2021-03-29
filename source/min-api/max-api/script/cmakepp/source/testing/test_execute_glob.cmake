function(test_execute_glob)
  timer_start(test_run)
  cd("${CMAKE_CURRENT_BINARY_DIR}")
  glob_ignore(${ARGN})
  ans(test_files)
  list(LENGTH test_files len)
  ## sort the test files so that they are always executed in the same order
  list(SORT test_files)
  message("found ${len} tests in path for '${ARGN}'")
  set(i 0)
  foreach(test ${test_files})
    math(EXPR i "${i} + 1")
    message(STATUS "test ${i} of ${len}")
    message_indent_push()
    test_execute("${test}")
    message_indent_pop()
    message(STATUS "done")
  endforeach()

  timer_print_elapsed(test_run)


endfunction()