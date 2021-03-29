## `()->{ <<identifier open>:<identifier close>...>... }`
##
## returns a map which contains all nesting pairs in cmake
function(cmake_ast_nesting_pairs)
  map_new()
  ans(nesting_start_end_pairs)
  map_set(${nesting_start_end_pairs} function endfunction)
  map_set(${nesting_start_end_pairs} while endwhile)
  map_set(${nesting_start_end_pairs} if elseif else endif)
  map_set(${nesting_start_end_pairs} elseif elseif else endif)
  map_set(${nesting_start_end_pairs} else endif)
  map_set(${nesting_start_end_pairs} macro endmacro)
  map_set(${nesting_start_end_pairs} foreach endforeach)
 
 return_ref(nesting_start_end_pairs)
endfunction()