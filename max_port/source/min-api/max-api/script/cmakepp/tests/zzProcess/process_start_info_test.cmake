function(test)
  define_test_function(test_uut process_start_info)



  # command
  test_uut("{command:'test'}" test)
  test_uut("{command:'test'}" "'test'")
  test_uut("{command:'test'}" "'test' a b c")
  test_uut("{command:'test'}" COMMAND test)
  test_uut("{command:'test'}" COMMAND test a b c)
  test_uut("{command:'test'}" "{path:'test'}")
  test_uut("{command:'test'}" "{path:'test', args:['asd','bsd']}")
  test_uut("{command:'test'}" "{command:'test'}")


  # working directory
  pushd(dir1 --create)
  ans(thepath)
  mkdir("dir2")
  mkdir("dir3")
  mkdir("dir4")
  
  message(inconclusive)
  return()
  test_uut("{ cwd: $thepath }" test)
  test_uut("{ cwd: $thepath }" COMMAND test)
  test_uut("{cwd:'${thepath}/dir2'}" test WORKING_DIRECTORY "dir2")
  test_uut("{cwd:'${thepath}/dir3'}" test WORKING_DIRECTORY "${thepath}/dir3")
  test_uut("{cwd:'${thepath}/dir4'}" "{command:'test', cwd:'dir4'}")
  test_uut("{cwd:'${thepath}/dir3'}" "{command:'test', cwd:'dir4'}" WORKING_DIRECTORY dir3)


  popd()


  # timeout
  test_uut("{timeout:-1}" test)
  test_uut("{timeout:3}" test TIMEOUT 3)
  test_uut("{timeout:4}" "{command:'test',timeout:4}")
  test_uut("{timeout:3}" "{command:'test',timeout:4}" TIMEOUT 3)



  # args

  test_uut("{args:'a'}" "test a")
  test_uut("{args:'a'}" COMMAND test a)
  test_uut("{args:'a'}" "{command:'test',args:['a']}")
  test_uut("{args:'a'}" "{
    command:'test',
    args:['a']
  }")


endfunction()