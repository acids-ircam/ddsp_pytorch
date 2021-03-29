function(test)
  

  pushd("dir1" --create)
  fwrite(text1.txt hello)
  popd()

  pushd("dir2" --create)
  fwrite(text1.txt hello)
  popd()

  pushd("dir3" --create)
  popd()

  fwrite(text1.txt hello)



  #file(MAKE_DIRECTORY "${test_dir}/dir1")
  #file(WRITE "${test_dir}/dir1/text1.txt" "hello")
  #file(MAKE_DIRECTORY "${test_dir}/dir2")
  #file(MAKE_DIRECTORY "${test_dir}/dir3")
  #file(WRITE "${test_dir}/dir2/text1.txt" "hello")
  #file(WRITE "${test_dir}/text1.txt" "hello")

  checksum_dir("${test_dir}")
  ans(chk)
  message("${chk}")

  file(WRITE "${test_dir}/test.txt" "asd")
  checksum_dir("${test_dir}")
  ans(chk)
  message("${chk}")
  file(REMOVE "${test_dir}/test.txt")
  checksum_dir("${test_dir}")
  ans(chk)
  message("${chk}")

  foreach(i RANGE 1000)
    checksum_layout("${test_dir}")


  endforeach()
endfunction()