function(test)

  fwrite("dir1/f1.txt" "asd")
  fwrite("dir1/dir2/f2.txt" "bsd")




  content_dir_check(dir1)
  ans(res)
  assert(${res} STREQUAL false )

  content_dir_update(dir1)
  ans(chk)
  assert(chk)

  content_dir_check(dir1)
  ans(res)
  assert(res)


  fwrite("dir1/f1.txt" "csd")

  content_dir_check(dir1)
  ans(res)
  assert("${res}" STREQUAL "false")

  content_dir_update(dir1)
  ans(chk)
  assert(chk)

  content_dir_check(dir1)
  ans(res)
  assert("${res}" STREQUAL "true")




endfunction()