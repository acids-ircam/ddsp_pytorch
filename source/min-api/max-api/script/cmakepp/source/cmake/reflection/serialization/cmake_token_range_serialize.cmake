## `(<start:<cmake token>> <end:<cmake token>>?)-><cmake code>`
## 
## generates the cmake code corresponding to the cmake token range
function(cmake_token_range_serialize range)
  cmake_token_range_to_list("${range}")
  ans(tokens)
  set(result)
  foreach(token ${tokens})
    map_tryget(${token} value)
    ans(value)
    set(result "${result}${value}")
  endforeach()

  return_ref(result)
endfunction()
