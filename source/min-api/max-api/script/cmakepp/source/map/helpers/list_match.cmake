

# matches the object list 
function(list_match __list_match_lst )
  map_matches("${ARGN}")
  ans(predicate)
  list_where("${__list_match_lst}" "${predicate}")
  return_ans()
endfunction()