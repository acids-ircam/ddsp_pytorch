## `(<clause map: <sequence>>)-> <cnf>`
##
##  
## 
## creates a conjunctive normal form from the specified input
## ```
## <cnf> ::= {
##   c_n : <uint>  # the number of clauses
##   c_last : <int>  # c_n - 1
##   clause_map : { <<clause index>:<clause>>... }
##   clause_atom_map : { <<clause index> : <atom index>... >...}
##   clause_literal_map : { <<clause index> : <literal index>...>...}
##   
##   a_n : <uint> # the number of atoms
##   a_last : <int>  # a_n - 1 
##   atom_map : { <<atom index>:<atom>>... }
##   atom_clause_map  : { <<atom index>:<clause index>...>...}
##   atom_literal_map :  {}
##   atom_literal_negated_map : {}
##   atom_literal_identity_map : {}
##   atom_index_map : {}
##   
##   l_n : <uint>
##   l_last : <int>
##   literal_map : {}
##   literal_atom_map : {}
##   literal_inverse_map : {}
##   literal_negated_map : {}
##   literal_index_map : {}
##   literal_clause_map : {}
## }
## ```
function(cnf clause_map)

  map_keys("${clause_map}")
  ans(clause_indices)

  sequence_new()
  ans(literal_map)
  sequence_new()
  ans(atom_map)
  sequence_new()
  ans(atom_literal_map)
  sequence_new()
  ans(literal_atom_map)
  map_new()
  ans(literal_index_map)
  map_new()
  ans(atom_index_map)
  sequence_new()
  ans(literal_negated_map)
  sequence_new()
  ans(literal_inverse_map)
  sequence_new()
  ans(atom_literal_negated_map)
  sequence_new()
  ans(atom_literal_identity_map)

  map_values(${clause_map})
  ans(tmp)
  set(literals)
  foreach(literal ${tmp})
    if("${literal}" MATCHES "^!?(.+)")
      list(APPEND literals ${CMAKE_MATCH_1})
    endif()
  endforeach()
  list_remove_duplicates(literals)

  foreach(literal ${literals})
      sequence_add(${atom_map} "${literal}")
      ans(ai)
      sequence_add(${literal_map} "${literal}")
      ans(li)
      sequence_add(${literal_map} "!${literal}")
      ans(li_neg)

      sequence_add(${literal_negated_map} false)
      sequence_add(${literal_negated_map} true)
      sequence_add(${atom_literal_map} ${li} ${li_neg})

      sequence_add(${literal_atom_map} ${ai})
      sequence_add(${literal_atom_map} ${ai})
      
      sequence_add(${atom_literal_negated_map} ${li_neg})
      sequence_add(${atom_literal_identity_map} ${li})

      sequence_add(${literal_inverse_map} ${li_neg})
      sequence_add(${literal_inverse_map} ${li})

      map_set(${literal_index_map} "${literal}" ${li})
      map_set(${literal_index_map} "!${literal}" ${li_neg})
      map_set(${atom_index_map} "${literal}" "${ai}")

  endforeach()

  map_new()
  ans(clause_atom_map)

  map_new()
  ans(clause_literal_map)

  map_new()
  ans(literal_clause_map)

  map_new()
  ans(atom_clause_map)

  foreach(ci ${clause_indices})
    map_tryget("${clause_map}" ${ci})
    ans(clause)
    map_set(${clause_atom_map} ${ci})
    map_set(${clause_literal_map} ${ci})
    foreach(literal ${clause})
      
      map_tryget(${literal_index_map} "${literal}")
      ans(li)

      map_tryget(${literal_atom_map} ${li})
      ans(ai)

      map_append_unique(${clause_atom_map} ${ci} ${ai})
      map_append_unique(${clause_literal_map} ${ci} ${li})
      map_append_unique(${literal_clause_map} ${li} ${ci})
      map_append_unique(${atom_clause_map} ${ai} ${ci})
    endforeach()
  endforeach()

  sequence_count(${clause_map})
  ans(c_n)
  math(EXPR c_last "${c_n} - 1")

  sequence_count(${literal_map})
  ans(l_n)
  math(EXPR l_last "${l_n} - 1")

  sequence_count(${atom_map})
  ans(a_n)
  math(EXPR a_last "${a_n} - 1")
  #json_print(${clause_map})

  map_capture_new(
    c_n
    c_last
    clause_map
    clause_atom_map
    clause_literal_map

    a_n
    a_last
    atom_map
    atom_clause_map
    atom_literal_map
    atom_literal_negated_map
    atom_literal_identity_map
    atom_index_map

    l_n 
    l_last
    literal_map
    literal_atom_map
    literal_inverse_map
    literal_negated_map
    literal_index_map
    literal_clause_map
  )
  ans(cnf)

  return_ref(cnf)

endfunction()