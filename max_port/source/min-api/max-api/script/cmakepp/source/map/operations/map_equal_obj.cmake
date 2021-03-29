## compares two maps for value equality
## lhs and rhs may be objectish 
function(map_equal_obj lhs rhs)
  obj("${lhs}")
  ans(lhs)
  obj("${rhs}")
  ans(rhs)
  map_equal("${lhs}" "${rhs}")
  return_ans()
endfunction()