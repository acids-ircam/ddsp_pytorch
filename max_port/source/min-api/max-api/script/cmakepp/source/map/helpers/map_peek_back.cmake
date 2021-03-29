
function(map_peek_back map prop)
  map_tryget("${map}" "${prop}")
  ans(lst)
  list_peek_back(lst)
  return_ans()
endfunction()