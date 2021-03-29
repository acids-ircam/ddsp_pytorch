function(require_map)
  map_set_hidden(:__require_map __type__ map)
  stack_new()
  ans(stack)
  map_set_hidden(:__require_map include_dirs ${stack})

  function(require_map)
    return(":__require_map")
  endfunction()
  require_map()
  return_ans()
endfunction()