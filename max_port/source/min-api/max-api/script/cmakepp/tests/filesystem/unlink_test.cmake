function(test)
  fwrite(dir2/f1.txt "hello")

  ln(dir2 dir1)

  assert(EXISTS "${test_dir}/dir1/f1.txt")
  assert(EXISTS "${test_dir}/dir2/f1.txt")

  unlink(dir1)

  assert(NOT EXISTS "${test_dir}/dir1/f1.txt")
  assert(EXISTS "${test_dir}/dir2/f1.txt")


  ln(dir2/f1.txt)

  assert(EXISTS "${test_dir}/f1.txt")

  unlink(f1.txt)

  #assert(NOT EXISTS "${test_dir}/f1.txt")



endfunction()