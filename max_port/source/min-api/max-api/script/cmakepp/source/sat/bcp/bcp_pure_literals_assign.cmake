## `(...)->...` 
##
## 
## assigns all pure literals (and there inverse) 
## removes all clauses an containing one from clauses
## returns all indices of pure literals
## returns conflict if a pure literal assignment conflicts with an existing one
function(bcp_pure_literals_assign f clauses assignments)
  bcp_pure_literals_find(${f} ${clauses})
  ans(pure_literals)

  if("${pure_literals}_" STREQUAL "_")
    return()
  endif()

  map_import_properties(${f} literal_inverse_map)

 # print_vars(assignments pure_literals)

  ## set assignments
  foreach(pure_literal ${pure_literals})
    bcp_assignment_add(${f} ${assignments} ${pure_literal} true)
    ans(ok)
    if(NOT ok)
      return(conflict)
    endif()

    map_tryget(${literal_inverse_map} ${pure_literal})
    ans(inverse)

    bcp_assignment_add(${f} ${assignments} ${inverse} false)
    ans(ok)
    if(NOT ok)
      return(conflict)
    endif()
  endforeach()

  ## remove clauses containing pure literal
  map_keys(${clauses})
  ans(clause_indices)

  foreach(ci ${clause_indices})
    map_tryget(${clauses} ${ci})
    ans(clause)

    list_contains_any(clause ${pure_literals})
    ans(contains_any)
  
    if(contains_any)
      map_remove(${clauses} ${ci})
    endif()      
  endforeach()
  return_ref(pure_literals)
endfunction()