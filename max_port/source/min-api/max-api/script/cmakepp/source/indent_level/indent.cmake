

  function(indent str)
    indent_get(${ARGN})
    ans(indent)
    set(str "${indent}${str}")
    return_ref(str)
  endfunction()
