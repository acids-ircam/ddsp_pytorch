
function(ast_new 
  tokens 
  expression_type 
  value_type 
  ref 
  code
  value 
  const 
  pure_value
  children

  )
  map_new()
  ans(ast)
  map_set(${ast} expression_type "${expression_type}")
  map_set(${ast} value_type "${value_type}")
  map_set(${ast} value "${value}")
  map_set(${ast} const "${const}")
  map_set(${ast} pure_value "${pure_value}")
  map_set(${ast} ref "${ref}")
  map_set(${ast} code "${code}")
  map_set(${ast} tokens "${tokens}")

  if(children)
    foreach(child ${children})
      map_set("${child}" parent "${ast}")
    endforeach()
    map_set("${ast}" children ${children})
  endif()

  return(${ast})
endfunction()
