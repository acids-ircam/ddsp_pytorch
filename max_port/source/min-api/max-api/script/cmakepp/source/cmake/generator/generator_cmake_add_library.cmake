function(test)

  fwrite("include/lib1.h" "")
  fwrite("include/lib2.h" "")
  fwrite("include/lib3.h" "")
  fwrite("include/lib4.h" "")
  fwrite("src/main.cpp" "")
  fwrite("src/impl1.cpp" "")
  fwrite("src/impl2.cpp" "")
  fwrite("src/impl3.cpp" "")
  fwrite("src/dir1/impl4.cpp" "")
  fwrite("src/dir2/impl5.cpp" "")

  
endfunction()


function(generator_cmake_source_group name)
  set(globs ${ARGN})
  glob_ignore(${globs} --relative)
  ans(files)


  set(template "## 


    ")

endfunction()


function(generator_cmake_add_library config)



endfunction()