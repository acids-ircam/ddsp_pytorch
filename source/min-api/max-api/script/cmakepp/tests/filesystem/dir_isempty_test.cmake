function(test)


  dir_isempty("${test_dir}")
  ans(res)
  assert(res)


  dir_isempty(.)
  ans(res)
  assert(res)

  fwrite("hello.txt" "hello")

  dir_isempty(".")
  ans(res)
  assert(NOT res)


  dir_isempty("${test_dir}")
  ans(res)
  assert(NOT res)


endfunction()