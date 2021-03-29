function(test)

  function(test_bcp assignments)

    data("${assignments}")
    ans(assignments)


    cnf("${arguments_sequence}")
    ans(f)

    map_tryget("${f}" clause_literal_map)
    ans(clauses)

    map_clone_deep("${clauses}")
    ans(clauses)

    timer_start(bcp)
    bcp("${f}" "${clauses}" "${assignments}")
    ans(result)
    timer_print_elapsed(bcp)

    cnf_print(${f})
    print_vars(result clauses assignments)

    map_capture_new(result assignments clauses)
    return_ans()

   endfunction()
 



 #  clauses: {"0":[1,2,4],"1":[2,5],"2":[1,6],"3":[4,7],"4":"0"} assignments: {"1":false}
 #  clauses: {} assignments: {"1":false,"0":true,"6":true,"7":false,"4":true,"5":false,"2":true,"3":false}
 # propagating !A = false => deduced: 0;1;6;7;4;5;2;3

  define_test_function(test_uut test_bcp assignments)

  test_uut("{}" "{}" "!a;b;c" "!b;!c" "!a;c" "!a;d" "c;!d" "a") # a:true c:true d:true b:true 
return()
  test_uut("{result:['0','1']}" "{}" a)
  test_uut("{result:['1','0']}" "{}" !a)
  test_uut("{result:['0','1','2','3']}" "{}" a b)
  test_uut("{result:['0','1','3','2']}" "{}" a !b)
  test_uut("{result:['0','1','2','3'], assignments:{'0':'true','1':'false', '2':'true', '3':'false'}}" "{}" a "!a;b")
  test_uut("{result:'conflict'}" "{}" a !a)
  test_uut("{result:'unsatisfied'}" "{'0':'true'}" "")
  test_uut("{result:null}" "{}")

endfunction()