
  macro(return_math expr)
    math(EXPR __return_math_res "${expr}")
    return(${__return_math_res})
  endmacro()
