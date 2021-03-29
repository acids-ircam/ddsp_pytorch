## `(...)->...` 
##
## convenience function
## same as cmake_token_range_filter however returns the token values
function(cmake_token_range_filter_values range)
  set(args ${ARGN})
  list_extract_flag(args --encode)
  ans(encode)## todo
  cmake_token_range_filter("${range}" ${args})
  ans(tokens)
  list_select_property(tokens literal_value)
  return_ans()
endfunction()
