## `()->`
##
## takes a set of clauses and simplifies them by 
## setting li to value
## removes all li that are false from clauses
## removes clauses which are satisfied
function(bcp_simplify_clauses f clauses li value)
 # map_import_properties(${f})

  map_keys(${clauses})
  ans(clause_indices)

  set(unit_literals)

  foreach(ci ${clause_indices})
    ## get clause's literal indices
    map_tryget(${clauses} ${ci})
    ans(clause)

    ## propagate new literal value to clause
    bcp_simplify_clause("${f}" "${clause}" "${li}" "${value}")
    ans(clause)

    if("${clause}_" STREQUAL "satisfied_")
      ## remove clause because it is always true
      map_remove("${clauses}" "${ci}")
    else()
      map_set(${clauses} ${ci} ${clause})
    endif()
  endforeach()
endfunction()