function(test)




  function(test_bcp_extract_unit_clauses)
    cnf(${arguments_sequence})
    ans(cnf)
    map_tryget(${cnf} clause_literal_map)
    ans(clauses)
   # cnf_print(${cnf})
    map_clone_deep("${clauses}")
    ans(clauses)

    timer_start(bcp_extract_unit_clauses)
    bcp_extract_unit_clauses("${cnf}" "${clauses}")
    ans(res)
    timer_print_elapsed(bcp_extract_unit_clauses)
    if("${res}" STREQUAL "unsatisfied")
      return(unsatisfied)
    endif()
    set(result)
    foreach(r ${res})
      assign(result[] = "cnf.literal_map.${r}")
    endforeach()

    #print_vars(result res clauses )
    return_ref(result)
  endfunction()

  define_test_function(test_uut test_bcp_extract_unit_clauses)

  test_uut("unsatisfied" "a" "" "c")
  test_uut("a" "a")
  test_uut("" "a;b")
  test_uut("c;e" "a;b;c" "c" "d;f" "e")
endfunction()