## `(<f:<cnf>> <clauses:{<<clause index>:<literal index...>...>}>)-><literal index...>`
##
## returns a list of literal indices of pure literals in clauses
function(bcp_pure_literals_find f clauses)
  map_import_properties(${f} literal_inverse_map)
  map_values(${clauses})
  ans(clause_literals)

  ## if all clauses are empty return nothing
  if("${clause_literals}_" STREQUAL "_")
    return()
  endif()

  ## loop through all literals of all clauses and check if its inverse was 
  ## not found append it to pure_literals (which are returned)
  list(REMOVE_DUPLICATES clause_literals)
  set(pure_literals)
  while(NOT "${clause_literals}_" STREQUAL "_")
    list_pop_front(clause_literals)
    ans(current_literal)

    map_tryget(${literal_inverse_map} ${current_literal})
    ans(inverse_literal)

    list(FIND clause_literals ${inverse_literal} inverse_found)

    if(${inverse_found} LESS 0)
      ## current literal is pure
      list(APPEND pure_literals ${current_literal})
    else()
      list(REMOVE_AT clause_literals ${inverse_found})
    endif()
  endwhile()

  return_ref(pure_literals)
endfunction()
