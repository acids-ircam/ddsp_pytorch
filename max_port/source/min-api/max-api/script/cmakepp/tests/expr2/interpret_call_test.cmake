function(test)




  set(exception "{'__$type__':'exception'}")

   function(my_func)
    arguments_encoded_list(0 ${ARGC})
    ans(result)

    list(INSERT result 0 my_func)
    #_message("${result}")
    return_ref(result)
  endfunction()
  set(thefunc my_func)

  string_codes()
  ##### runtime tests #####

  ## interpret call tests
  define_test_function2(test_uut expr_eval interpret_call "")


  test_uut("${exception}")  # no token
  test_uut("${exception}" "")  # empty token
  test_uut("${exception}" "abc")  # no paren -> no call 
  test_uut("${exception}" "()")  # no lhs rvalue
  test_uut("${exception}" "()()") ## illegal lhs rvalue  
 

  ## static functions (calls the function directly)

  ## static function single const parameter
  test_uut("my_func;1" "my_func(1)")
  ## static function empty parameters
  test_uut("my_func" "my_func()")  
  ## stati function multiple const parameters
  test_uut("my_func;1;2" "my_func(1,2)")
  ## static function 3 const parameters
  test_uut("my_func;1;2;3" "my_func(1,2,3)")

  ## static function multiple const arguments, no spread 
  test_uut("my_func;1;2${semicolon_code}3;4" "my_func(1,[2,3],4)") 
  ## static function non const arguments, no spread 
  test_uut("my_func;my_func${semicolon_code}my_func" "my_func(my_func(my_func()))") 

  ## static function multiple arguments with spread (ellipsis `...`)
  test_uut("my_func;1;2;3;4" "my_func(1,[ 2 , 3]...,4)")
  

  ## dynamic functions (uses eval to call function)

  ## dynamic function no arguments
  test_uut("my_func" "$thefunc()")
  ## dynamic function single const argument
  test_uut("my_func;1" "$thefunc(1)")
  ## dynamic function multiple const arguments, no spread
  test_uut("my_func;1${semicolon_code}2;3${semicolon_code}4" "$thefunc([1,2],[3,4])")

  ## unqoted string arguments
  test_uut("my_func;hello" "$thefunc(hello)")
  test_uut("my_func;helloworld" "$thefunc(hello world)")
  test_uut("my_func;hello;world" "$thefunc(hello,world)")
  test_uut("my_func;hello world" $thefunc( "hello world" ))
  test_uut("my_func;hello world;hello person" $thefunc( "hello world" , "hello person" ))
  
  ## single  quoted string arguments
  test_uut("my_func;\\" "$thefunc('\\')")
  test_uut("my_func;\\" $thefunc('\\'))





endfunction()