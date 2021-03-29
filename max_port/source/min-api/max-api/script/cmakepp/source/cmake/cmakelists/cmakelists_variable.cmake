## `(<cmakelists> <variable path>)-><any>...`
## 
## see list_modify
## modifies a variable returns the value of the variable
function(cmakelists_variable cmakelists variable_path)
  map_tryget(${cmakelists} begin)
  ans(range)
  cmake_token_range_variable_navigate("${range}" "${variable_path}" ${ARGN})
  return_ans()
endfunction()