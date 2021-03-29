function(test)

  



  mkdir("dir1")
  mkdir("dir11")
  fwrite("f11.txt" "asd")
  fwrite("f12.txt" "asd")

  mkdir(dir2)
  pushd("dir2")
  fwrite("f21.txt" "asd")
  fwrite("f22.txt" "asd")

  mkdir(dir3)
  pushd(dir3)
  fwrite("f31.txt" "asd")
  fwrite("f32.txt" "asd")

  glob_up(2 *.txt)
  ans(res_gl)


  popd()
  popd()


  glob("${test_dir}/**" --recurse)
  ans(res)




endfunction()