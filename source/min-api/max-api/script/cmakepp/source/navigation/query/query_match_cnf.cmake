##  `(<clauses: <clause: { <selector>:<literal...> }>...> <any...>)-><bool>`
## 
##  queries the specified args for the specified clauses in conjunctive normal form
function(query_match_cnf clauses)
  data("${clauses}")
  ans(clauses)

  foreach(clause ${clauses})
    query_disjunction(${clause} ${ARGN})
    ans(clause_result)
    if(NOT clause_result)
      return(false)
    endif()
  endforeach()
  return(true)
endfunction()
