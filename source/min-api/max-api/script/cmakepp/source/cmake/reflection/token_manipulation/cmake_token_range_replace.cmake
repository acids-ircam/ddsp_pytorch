## `(<range:<cmake token range>> <replace_range:<cmake token range>>)-><cmake token range>`
## 
## replaces the specified range with the specified replace range
## returns the replace range
function(cmake_token_range_replace range replace_range)
  cmake_token_range("${range}")
  ans_extract(start end)
  cmake_token_range("${replace_range}")
  ans_extract(replace_start replace_end)
  map_tryget(${start} previous)
  ans(previous)
  map_set(${previous} next ${replace_start})
  map_set(${replace_start} previous ${previous})
  map_set(${end} previous ${replace_end})
  map_set(${replace_end} next ${end})
  return(${replace_start} ${replace_end})
endfunction()


