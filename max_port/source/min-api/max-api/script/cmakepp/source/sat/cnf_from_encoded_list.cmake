
  function(cnf_from_encoded_list)
    arguments_sequence(0 ${ARGC})
    ans(clauses)
    cnf("${clauses}")
    return_ans()
  endfunction()