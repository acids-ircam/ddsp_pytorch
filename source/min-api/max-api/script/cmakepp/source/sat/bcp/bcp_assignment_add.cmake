## `()->`
##
## tries to add a assignment for literal li
## if the assignment does not exist it is set and true is returned
## if an assignment exists and it conflicts with the new assignemnt return false
## if the assignment exists and is equal to the new assignment nochange is returned
## if(result) => ok
## if(NOT result) => conflict
function(bcp_assignment_add f assignments li value)
#  print_vars(assignments li value)
  map_tryget("${assignments}" ${li})
  ans(existing_value)
  if("${existing_value}_" STREQUAL "_")
    map_set(${assignments} ${li} ${value})
    return(true)
  elseif(NOT "${existing_value}" STREQUAL "${value}")
    return(false)
  endif()
  return(nochange)
endfunction()