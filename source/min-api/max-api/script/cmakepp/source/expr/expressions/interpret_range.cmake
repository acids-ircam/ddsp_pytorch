

function(interpret_range tokens)
  interpret_literal("${tokens}")
  ans(index)
  map_tryget("${index}" value_type)
  ans(value_type)
  if(NOT "${value_type}" STREQUAL "number")
    throw("invalid range")
  endif()

  map_tryget("${index}" value)
  ans(value)
  set(code)
  ast_new(
    "${tokens}"
    "range"
    "range"
    ""
    "${code}"
    "${value}"
    "true"
    "true"
    ""
    )
  ans(ast)
  return_ref(ast)
endfunction()


