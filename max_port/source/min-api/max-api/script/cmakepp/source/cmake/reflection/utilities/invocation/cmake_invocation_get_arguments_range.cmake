## `(<invocation:<command invocation>>)->[<start:<token>> <end:<token>>]`
## 
## returns the token range of the invocations arguments given an invocation token
function(cmake_invocation_get_arguments_range invocation)
  cmake_token_range_find_next_by_type("${invocation}" nesting)
  ans(arguments_begin)
  map_tryget(${arguments_begin} end)
  ans(arguments_end)
  map_tryget(${arguments_begin} next)
  ans(arguments_begin)
  return(${arguments_begin} ${arguments_end})
endfunction()
