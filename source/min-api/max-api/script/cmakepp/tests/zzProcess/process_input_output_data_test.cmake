function(test)

  ## stdout and return_code is correct?
  process_start_script("message(STATUS hello)")
  ans(handle)

  process_wait(${handle})
  process_return_code(${handle})
  ans(retcode)
  assert("${retcode}" EQUAL 0)
  process_stdout(${handle})
  ans(out)
  assert("${out}" MATCHES "hello")
  process_stderr(${handle})
  ans(err)
  assert("${err}_" STREQUAL "_")

  ## stderr and stdout and return code correct?
  process_start_script("message(STATUS hello)\nmessage(FATAL_ERROR ohno)")
  ans(handle)
  process_wait(${handle})

  process_return_code(${handle})
  ans(retcode)

  process_stderr(${handle})
  ans(err)

  process_stdout(${handle})
  ans(out)

  assert("${out}" MATCHES "hello")
  assert("${err}" MATCHES "ohno")
  assert(NOT "${retcode}" EQUAL "0")


  ## command line argument passed to script
  process_start_script("message(STATUS \${CMAKE_ARGV3})" gagagugu)
  ans(handle)

  process_wait(${handle})
  process_stdout(${handle})
  ans(stdout)
  assert("${stdout}" MATCHES "gagagugu")


  ## env variable passed to script
  set(ENV{customvar} kakakaka)
  process_start_script("message(STATUS \$ENV{customvar})" )
  ans(handle)

  process_wait(${handle})
  process_stdout(${handle})
  ans(stdout)
  assert("${stdout}" MATCHES "kakakaka")


endfunction()