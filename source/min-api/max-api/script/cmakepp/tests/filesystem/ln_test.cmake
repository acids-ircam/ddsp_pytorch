function(test)

  
  fwrite("dir1/test.txt" "abc")

  ln("dir1/test.txt" "kaka.txt")

  fread("kaka.txt")
  ans(res)

  assert("${res}" STREQUAL "abc")

  fwrite("kaka.txt" "ksd")
  fread("dir1/test.txt")
  ans(res)
  assert("${res}" STREQUAL "ksd")


  fwrite("dir1/test.txt" abc)
  fread("kaka.txt")
  ans(res)
  assert("${res}" STREQUAL "abc")

  ln("dir1" "dir2")
  ans(res)
  assert(res)

  assert(EXISTS "${test_dir}/dir1/test.txt")
  assert(EXISTS "${test_dir}/dir2/test.txt")

  fread("${test_dir}/dir2/test.txt")
  ans(res)
  assert("${res}" STREQUAL "abc")


  fwrite("${test_dir}/dir1/test.txt" "cde")

  fread("${test_dir}/dir2/test.txt")
  ans(res)
  assert("${res}" STREQUAL "cde")

  

  fwrite("${test_dir}/dir2/test.txt" "ekd")

  fread("${test_dir}/dir1/test.txt")
  ans(res)
  assert("${res}" STREQUAL "ekd")


  ln("dir1/test.txt")

  assert(EXISTS "${test_dir}/test.txt")
  fread("${test_dir}/test.txt")
  ans(res)

  assert("${res}" STREQUAL "ekd")



endfunction()