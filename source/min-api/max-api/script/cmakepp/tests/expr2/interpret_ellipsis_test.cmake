function(test)

  set(exception "{'__$type__':'exception'}")

  ##### compile time tests ######



  define_test_function2(test_uut expr_parse interpret_ellipsis "")

  ## no tokens fails
  test_uut("${exception}")
  ## wrong tokens fails
  test_uut("${exception}" 1 2 3 4)
  ## no rvalue 
  test_uut("${exception}" "...")
  ## illegal rvalue 
  test_uut("${exception}" ",...")
  ## ok
  test_uut("{expression_type:'ellipsis', children:{value:'a'}}" "a...")
  ## ok
  test_uut("{expression_type:'ellipsis', children:{children:[{value:'a'},{value:'b'}]}}" "[a,b]...")


endfunction()