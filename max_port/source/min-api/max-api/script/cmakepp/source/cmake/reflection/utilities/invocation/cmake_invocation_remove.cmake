## `(<cmake invocation>)-><void>`
##
## removes the specified invocation from its context by removing the invocation token and the arguments from the linked list that they are part
function(cmake_invocation_remove invocation)
  map_tryget(${invocation} invocation_token)
  ans(begin)
  map_tryget(${invocation} arguments_end_token)
  ans(end)
  cmake_token_range_remove("${begin};${end}")
  return()
endfunction()

