## `(<f:<cnf>> <clause:<literal index...>> <li:literal index> <value:<bool>>)-><literal index..>|"satisfied"`
##
## returns `"satisfied"` if clause is satisfied by literal assignment
## returns `<null>` if clause is unsatisfiable
## returns clause with `<li>` removed if `<value>` is false
function(bcp_simplify_clause f clause li value)
  list(FIND clause ${li} found)

  if("${found}" LESS 0)
    ## literal not found in clause -> no change 
    ## if clause was unsatisfied it stays unsatisfied
    return_ref(clause)
  endif()

  if(value)
    ## literal is in clause and is true => clause is satisfied
    return(satisfied)
  endif()

  ## if clause is not unsatisfied
  ## remove false value from clause as it does not change the result of clause
  if(clause)
    list(REMOVE_ITEM clause ${li})
  endif()

  ## return rest of clause
  ## if clause was unsatisfied it stays unsatisfied
  return_ref(clause)
endfunction()