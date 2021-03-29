function(test)


  path_parent_dirs("path/dir1/dir2/file.ext" 3)
  ans(dirs)
  
  pwd()
  ans(cwd)



  assert(${dirs} EQUALS "${cwd}/path/dir1/dir2" "${cwd}/path/dir1" "${cwd}/path")


endfunction()