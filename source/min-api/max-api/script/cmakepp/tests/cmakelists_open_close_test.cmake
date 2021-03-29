function(test)



  cmakelists_open()
  ans(cmakelists)
  assert(NOT cmakelists)


  fwrite(CMakeLists.txt)
  cmakelists_open()
  ans(cmakelists)

  rm(CMakeLists.txt)


  cmakelists_new("")
  ans(cmakelists)

  cmakelists_close(${cmakelists})
  assert(EXISTS CMakeLists.txt)


  pushd(--create dir1)
  cmakelists_open()
  ans(cmakelists)
  assertf({cmakelists.path} STREQUAL "${test_dir}/CMakeLists.txt")



endfunction()