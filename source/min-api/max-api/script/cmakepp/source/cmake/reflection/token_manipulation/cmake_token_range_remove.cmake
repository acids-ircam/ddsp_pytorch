## `(<cmake token range>)-><void>`
##
## removes the specified token range from the linked list
function(cmake_token_range_remove range)
  list_extract(range begin end)
  map_tryget("${begin}" previous)
  ans(before)
  map_tryget("${end}" next)
  ans(after)
  map_set("${before}" next ${after})
  map_set("${after}" previous ${before})
  return()
endfunction()

