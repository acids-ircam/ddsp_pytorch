
  macro(return_nav)
    assign(result = ${ARGN})
    return_ref(result)
  endmacro()