#returns the last returned value
# this is a shorthand useful when returning the rsult of a previous function
macro(return_ans)
  return_ref(__ans)
endmacro()
