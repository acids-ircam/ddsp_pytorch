function(test)


  ## runtime tests
  obj("{
    a:{
      b:{
        c:[1,2]
        },
      d:[3,4]
    },
    e:[
      {a:5},
      {a:6}
    ],
    f:7,
    g:8
  }")
  ans(the_object)

  data("[{a:1},{a:2}]")
  ans(the_object_list)



  define_test_function2(test_uut expr_eval "interpret_indexation" "")

  set(the_list a b c)

  test_uut("" "$the_object_list['a']")
  test_uut("1;2" "$the_object_list...['a']")

  ## property indexation, multi index indexation, multi property select
  test_uut("5;6" "$the_object.e[0,1]...['a']")
  ## successive property indexation
  test_uut("3;4" "$the_object['a']['d']")
  ## successive property and mulit index indexation
  test_uut("4;3" "$the_object['a']['d'][1,0]")
  ## proeprty indexation
  test_uut("7" "$the_object['f']")
  ## multi property indexation
  test_uut("7;8" "$the_object['f','g']")
  ## multi property indexation
  test_uut("2;1" "{a:1,b:2}['b','a']")
  ## mulit index indexation
  test_uut("c;b;d;a" "[a,b,c,d][2,1,3,0]")
  ## single index indexation
  test_uut("b" "$the_list[1]")
  ## mulit index indexation
  test_uut("c;a;b" "$the_list[2,0,1]")


  set(exception "{'__$type__':'exception'}")
  ##### compile time tests #####


  define_test_function2(test_uut expr_parse "interpret_indexation" "")

  ## invalid token count
  test_uut("${exception}")
  ## wrong token
  test_uut("${exception}" a)
  ## empty
  test_uut("{expression_type:'indexation'}" "a[]")
  ## simple string literal
  test_uut("{expression_type:'indexation'}" "a['abc']")
  ## number literal
  test_uut("{expression_type:'indexation'}" "a[1]")




endfunction()