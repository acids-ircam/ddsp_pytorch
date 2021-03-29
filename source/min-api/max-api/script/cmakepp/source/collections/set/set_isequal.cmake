# retruns true iff lhs and rhs are the same set (ignoring duplicates)
# the null set is only equal to the null set 
# the order of the set (as implied in being a set) does not matter
function(set_isequal __set_equal_lhs __set_equal_rhs)
  set_issubset(${__set_equal_lhs} ${__set_equal_rhs})
  ans(__set_equal_lhsIsInRhs)
  set_issubset(${__set_equal_rhs} ${__set_equal_lhs})
  ans(__set_equal_rhsIsInLhs)
  if(__set_equal_lhsIsInRhs AND __set_equal_rhsIsInLhs)
    return(true)
  endif() 
  return(false)
endfunction()