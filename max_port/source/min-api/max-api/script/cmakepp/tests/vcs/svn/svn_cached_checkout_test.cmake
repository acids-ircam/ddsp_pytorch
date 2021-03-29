function(test)

  svn_cached_checkout("https://github.com/toeb/test_repo@2" co1)
  ans(res)
  assert("${res}" STREQUAL "${test_dir}/co1")
  assert(EXISTS "${res}/package.cmake")


  svn_cached_checkout("https://github.com/toeb/test_repo@1" co2)
  ans(res)
  assert("${res}" STREQUAL "${test_dir}/co2")
  assert(NOT EXISTS "${res}/package.cmake")
  assert(EXISTS "${res}/README.md")

return()
  timer_start(timer1)
  svn_cached_checkout("https://github.com/toeb/test_repo@2" "svnco1" --refresh) 
  timer_print_elapsed(timer1)

  timer_start(timer1)
  svn_cached_checkout("https://github.com/toeb/test_repo@2" "svnco1")
  timer_print_elapsed(timer1)


  timer_start(timer1)
  svn_cached_checkout("https://github.com/toeb/test_repo@2" "svnco1" --readonly)
  timer_print_elapsed(timer1)

endfunction()