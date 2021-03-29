## `()->`
##
## returns unsatisfied if a clause is unsatisfieable
## returns indices of unit_clauses' literal
## else returns nothing 
## sideffect updates clauses map removes unit clauses
function(bcp_extract_unit_clauses f clauses)
 # map_import_properties(${f})
  map_keys(${clauses})
  ans(clause_indices)

  set(unit_literals)

  foreach(ci ${clause_indices})
    # ## get clause's literal indices
    map_tryget(${clauses} ${ci})
    ans(clause)

    if("${clause}_" STREQUAL "_")
      return(unsatisfied)
    endif()

    ## check if clause has become unit
    list(LENGTH clause literal_count)
    if("${literal_count}" EQUAL 1)
      ## if so remove it and collect the unit literal
      map_remove(${clauses} ${ci})
      list(APPEND unit_literals ${clause})
    else()
      ## update clause 
      map_set(${clauses} ${ci} ${clause})
    endif()

  endforeach()
  return_ref(unit_literals)
endfunction()
