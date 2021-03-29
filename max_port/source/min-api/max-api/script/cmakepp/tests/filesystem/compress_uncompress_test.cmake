function(test)

  mkdir("${test_dir}")
  cd("${test_dir}")

  fwrite(myfile1.txt "hello")
  fwrite(myfile2.txt "hello")
  fwrite(myfile3.txt "hello")
  fwrite(asd/myfile4.txt "asd")

  tar_lean("cvzf" "${test_dir}/myfile.tgz" "${test_dir}/asd/myfile4.txt" "myfile1.txt" "myfile2.txt" "myfile3.txt")

  assert(EXISTS "${test_dir}/myfile.tgz")


  compress("myfile2.tgz" "*.txt" --recurse)

  mkdir(res)
  pushd(res)

  uncompress(../myfile2.tgz)


  assert(EXISTS "${test_dir}/res/myfile1.txt")
  assert(EXISTS "${test_dir}/res/myfile2.txt")
  assert(EXISTS "${test_dir}/res/myfile3.txt")
  assert(EXISTS "${test_dir}/res/asd/myfile4.txt")



  compress("myfile3.tgz" "${test_dir}/myfile1.txt" "${test_dir}/myfile2.txt")

  assert(EXISTS "${test_dir}/res/myfile3.tgz")
endfunction()