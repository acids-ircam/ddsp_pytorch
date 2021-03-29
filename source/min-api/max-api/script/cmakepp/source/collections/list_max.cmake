## returns the maximum value in the list 
## using the specified comparerer function
function(list_max lst comparer)
  list_fold(${lst} "${comparer}")
  ans(res)
  return(${res})
endfunction()
