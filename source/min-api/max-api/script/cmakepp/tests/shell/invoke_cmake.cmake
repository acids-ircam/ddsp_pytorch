function(test)
  mkdir("${test_dir}")
  cd("${test_dir}")
  cmake(--help-command string --process-handle)
  ans(res)

  map_tryget(${res} exit_code)
  ans(error)

  assert(NOT error)
endfunction()