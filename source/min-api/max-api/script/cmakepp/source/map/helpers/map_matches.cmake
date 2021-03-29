
# returns a function which returns true of all 
function(map_matches attrs)
  obj("${attrs}")
  ans(attrs)
#  curry(map_match_properties(/1 ${attrs}))
  curry3(map_match_properties(/0 ${attrs}))
  return_ans()
endfunction()


