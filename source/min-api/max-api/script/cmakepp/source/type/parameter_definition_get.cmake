

function(parameter_definition_get name)
  map_tryget(__global_definitions "${name}")
  return_ans()
endfunction()

