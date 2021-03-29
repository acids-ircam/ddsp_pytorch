# returns true iff lhs is subset of rhs
# duplicate elements in lhs and rhs are ignored
# the null set is subset of every set including itself
# no other set is subset of the null set
# if rhs contains all elements of lhs then lhs is the subset of rhs
function(set_issubset __set_is_subset_of_lhs __set_is_subset_of_rhs)
  list(LENGTH ${__set_is_subset_of_lhs} __set_is_subset_of_length)
  if("${__set_is_subset_of_length}" EQUAL "0")
    return(true)
  endif()
  list(LENGTH ${__set_is_subset_of_rhs} __set_is_subset_of_length)
  if("${__set_is_subset_of_length}" EQUAL "0")
    return(false)
  endif()
  foreach(__set_is_subset_of_item ${${__set_is_subset_of_lhs}})
    list(FIND ${__set_is_subset_of_rhs} "${__set_is_subset_of_item}" __set_is_subset_of_idx)
    if("${__set_is_subset_of_idx}" EQUAL "-1")
      return(false)
    endif()
  endforeach()
  return(true)
endfunction()

