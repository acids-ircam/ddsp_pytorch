function(test)



  path_split("c:/dir1/dir2/file.ext")
  ans(res)
  assert(EQUALS ${res} c: dir1 dir2 file.ext)

  path_split("")
  ans(res)
  assert(NOT res)

  path_split("/test/test/test")
  ans(res)
  assert(EQUALS ${res} test test test)


  path_split("test")
  ans(res)
  assert(EQUALS ${res} test)



endfunction()