function(test)



  glob_expression_parse(asd.*)
  ans(res)
  

  glob_expression_parse(asd.* bsd.*)
  ans(res)
  

  glob_expression_parse(!asd.*)
  ans(res)

  glob_expression_parse(!asd.* !bsd.*)
  ans(res)


  glob_expression_parse(!asd.*  csd.*  !bsd.* dsd.*)
  ans(res)



  fwrite("hello.txt" "asd")
  fwrite("dir1/hello2.txt" "asd")
  fwrite("dir2/hello3.txt" "asd")
  fwrite("dir3/hello4.txt" "asd")
  fwrite("dir3/dir4/hello5.txt" "asd")


  glob_ignore(** !hello4.txt !hello5.txt --relative)
  ans(res)

  cp_glob(tgt **.txt  --recurse)



  return()
  
endfunction()