
  ## overwrites all values of lhs with rhs
  function(map_overwrite lhs rhs)
    obj("${lhs}")
    ans(lhs)
    obj("${rhs}")
    ans(rhs)

    map_iterator("${rhs}")
    ans(it)
    while(true)
      map_iterator_break(it)
      map_set("${lhs}" "${it.key}" "${it.value}")
    endwhile()
    return(${lhs})
  endfunction()
