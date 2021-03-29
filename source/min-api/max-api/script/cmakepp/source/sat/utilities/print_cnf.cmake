
  function(print_cnf f)
    scope_import_map(${f})
    print_multi(${c_last} clauses clause_literals clause_atoms)
    print_multi(${a_last} atoms atom_literals atom_clauses)
    print_multi(${l_last} literals literal_inverse literal_negated literal_clauses literal_atom)

  endfunction()

  ## new
  function(cnf_print f)
    scope_import_map(${f})
    print_multi(${c_last} clause_map clause_literal_map clause_atom_map)
    print_multi(${a_last} atom_map atom_literal_map atom_clause_map)
    print_multi(${l_last} literal_map literal_inverse_map literal_negated_map literal_clause_map literal_atom_map)

  endfunction()