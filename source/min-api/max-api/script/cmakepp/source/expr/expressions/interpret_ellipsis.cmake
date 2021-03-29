
function(interpret_ellipsis tokens)
  set(dots)
  list_pop_back(tokens)
  ans_append(dots)
  list_pop_back(tokens)
  ans_append(dots)
  list_pop_back(tokens)
  ans_append(dots)

  list_select_property(dots type)
  ans(dot_types)

  if(NOT "${dot_types}" STREQUAL "dot;dot;dot")    
    throw("not ellipsis: ${dot_types}")
  endif()

  if(NOT tokens)
    throw("no left hand rvalue")
  endif()
  

  interpret_rvalue("${tokens}")
  rethrow()
  ans(rvalue)




  map_tryget("${rvalue}" ref)
  ans(rvalue_ref)

  map_tryget("${rvalue}" value_type)
  ans(rvalue_value_type)

  map_tryget("${rvalue}" value)
  ans(rvalue_value)

  map_tryget("${rvalue}" const)
  ans(rvalue_const)

  map_tryget("${rvalue}" pure_value)
  ans(rvalue_pure_value)


  ast_new(
    "${tokens}"
    ellipsis
    "${rvalue_value_type}"
    "${rvalue_ref}"
    ""                      # code
    "${rvalue_value}"
    "${rvalue_const}"
    "${rvalue_pure_value}"
    "${rvalue}"
    )


  ans(ast)
  return_ref(ast)
endfunction()