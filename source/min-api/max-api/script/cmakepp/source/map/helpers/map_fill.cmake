

  ## files non existing or null values of lhs with values of rhs
  function(map_fill lhs rhs)
    map_ensure(lhs rhs)
    map_iterator(${rhs})
    ans(it)
    while(true)
      map_iterator_break(it)
    
      map_tryget(${lhs} "${it.key}")
      ans(lvalue)

      if("${lvalue}_" STREQUAL "_")
        map_set(${lhs} "${it.key}" "${it.value}")
      endif()
    endwhile()
    return_ref(lhs)
  endfunction()