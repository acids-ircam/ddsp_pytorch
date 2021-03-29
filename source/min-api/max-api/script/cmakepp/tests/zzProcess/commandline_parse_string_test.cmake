function(test)




  define_test_function(test_uut command_line_parse_string)

  test_uut("{command:'a b'}" "\"a b\" c d e")
  test_uut("{command:'a'}" "a b c")
return()
  
  test_uut(
    "{command:'a b', args:['c','d','e']}"
    "\"a b\" c d e")

  test_uut(
    "{command:'test', args:['a','b','c']}" 
    "test a b c"
    )
  test_uut(
    "{command:'c:/my/path/to/command.exe', args:['a b', 'c','d']}"
    "c:\\my\\path\\to\\command.exe \"a b\" c d"
    )  

  test_uut(
    "{command:'../asd', args:[
      'ab',
      'bc',
      'c d',
      'e',
      'lll ooko',
      'f',
      'g'

    ]}"
    "../bsd/../asd ab bc \"c d\" e \"lll ooko\" f g"
    )



endfunction()