## `(target_name:<regex>)-><cmake target>`
##
## returns all targets whose name match the specified regular expression
function(cmakelists_targets cmakelists target_name )
  map_tryget(${cmakelists} range)
  ans(range)
  cmake_token_range_targets_filter("${range}" "${target_name}")
  return_ans()
endfunction()