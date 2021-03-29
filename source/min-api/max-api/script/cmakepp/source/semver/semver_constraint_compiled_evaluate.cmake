
function(semver_constraint_compiled_evaluate compiled_constraint version )
  map_import_properties(${compiled_constraint} elements template)

  foreach(element ${elements})
    semver_constraint_evaluate_element("${element}" "${version}")
    ans(res)
    string(REPLACE "${element}" "${res}" template "${template}")
  endforeach()

  if(${template})
    return(true)
  endif()
  return(false)
endfunction()