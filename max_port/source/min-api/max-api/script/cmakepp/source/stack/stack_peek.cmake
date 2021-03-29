
  function(stack_peek stack)
    map_tryget("${stack}" back)
    ans(back)
    map_tryget("${stack}" "${back}")
    return_ans()
  endfunction()