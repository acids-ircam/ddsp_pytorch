function(test)
  git_repository_name("https://github.com/AnotherFoxGuy/cmakepp.git")
  ans(res)
  assert("${res}" STREQUAL "cmakepp")

  git_repository_name("${test_dir}/dir1")
  ans(res)
  assert("${res}" STREQUAL "dir1")



endfunction()