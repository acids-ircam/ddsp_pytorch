
  ## returns the value at idx
  function(map_at map idx)
    map_key_at(${map} "${idx}")
    ans(key)
    map_tryget(${map} "${key}")
    return_ans()
  endfunction()