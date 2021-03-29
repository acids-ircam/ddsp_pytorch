
  function(require_include_dirs )
    require_map()
    ans(map)
    map_get(${map}  include_dirs)
    ans(stack)
    stack_pop(${stack})
    ans(dirs)
    list(APPEND dirs ${ARGN})
    stack_push(${stack} ${dirs})

  endfunction()
