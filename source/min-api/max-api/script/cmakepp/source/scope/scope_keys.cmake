

# returns all currently defined variables of the local scope
function(scope_keys)
  get_cmake_property(_variableNames VARIABLES)
  return_ref(_variableNames)
endfunction()