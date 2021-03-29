## `(<where:<cmake token>> <cmake token range> )-><token range>`
##
## inserts the specified token range before <where>
function(cmake_token_range_insert where what)
  cmake_token_range("${what}")
  ans_extract(begin end)
  map_tryget("${where}" previous)
  ans(previous)

  if(previous)
    map_set_hidden(${previous} next ${begin})
    map_set_hidden(${begin} previous ${previous})  
  endif()
  map_set_hidden(${end} next ${where})
  map_set_hidden(${where} previous ${end}) 

  return(${begin} ${end})
endfunction()