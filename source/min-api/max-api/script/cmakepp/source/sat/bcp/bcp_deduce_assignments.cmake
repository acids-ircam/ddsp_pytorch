## `()->`
##
## takes a list of clauses and deduces all assignments from unit clauses
## storing them in assignments and returning their literal indices
## returns conflict if a deduced assignment conflicts with an existing assignment
## return unsatisfied if clauses contains at least one unsatisfiable clause
function(bcp_deduce_assignments f clauses assignments)
    map_import_properties(${f} literal_inverse_map)
    bcp_extract_unit_clauses("${f}" "${clauses}")
    ans(unit_clauses)


    if("${unit_clauses}" MATCHES "unsatisfied")
      return(unsatisfied)
    endif()

    set(deduced_assignments)
    foreach(unit_clause ${unit_clauses})
      bcp_assignment_add("${f}" "${assignments}" "${unit_clause}" true)
      ans(ok)
      if(NOT ok)
        return(conflict)
      endif()

      map_tryget(${literal_inverse_map} ${unit_clause})
      ans(unit_clause_inverse)

     # print_vars(unit_clause unit_clause_inverse)

      bcp_assignment_add("${f}" "${assignments}" "${unit_clause_inverse}" false)
      ans(ok)
    #  print_vars(ok)
      if(NOT ok)
        return(conflict)
      endif()
      #  messaGE(FORMAT "  deduced {f.literal_map.${unit_clause}} to be true ")
      #  messaGE(FORMAT "  deduced {f.literal_map.${unit_clause_inverse}} to be false ")

      list(APPEND deduced_assignments ${unit_clause}  ${unit_clause_inverse} )
    endforeach()
    list_remove_duplicates(deduced_assignments)
    return_ref(deduced_assignments)
endfunction()