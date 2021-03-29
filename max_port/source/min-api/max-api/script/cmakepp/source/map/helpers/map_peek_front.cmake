
function(map_peek_front map prop)
  map_tryget("${map}" "${prop}")
  ans(lst)
  list_peek_front(lst)
  return_ans()
endfunction()