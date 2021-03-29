function(test)

  function(test_bcp_deduce_assignments assignments)
    data("${assignments}")
    ans(assignments)

    cnf(${arguments_sequence})
    ans(cnf)

    map_tryget(${cnf} clause_literal_map)
    ans(clauses)
    map_clone_deep(${clauses})
    ans(clauses)

#   cnf_print(${cnf})
   timer_start(bcp_deduce_assignments)
    bcp_deduce_assignments("${cnf}" "${clauses}" "${assignments}")
    ans(result)
   timer_print_elapsed(bcp_deduce_assignments)

    map_capture_new(result assignments)
    return_ans()
  endfunction()

  define_test_function(test_uut test_bcp_deduce_assignments assignments)

  test_uut("{assignments:{}, result:null}" "{}" )
  test_uut("{assignments:{'0':'true','1':'false'}, result:['0','1']}" "{}" 0)
  test_uut("{assignments:{'0':'false', '1':'true'}, result:['1','0']}" "{}" !0)
  test_uut("{result:'conflict'}" "{'0':'false'}" 0)
  test_uut("{result:'conflict'}" "{}" 0 !0)

endfunction()