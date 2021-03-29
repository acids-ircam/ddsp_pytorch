# clears the current local scope of any variables
function(scope_clear)
  scope_keys()
  ans(vars)
  foreach (var ${vars})
    set(${var} PARENT_SCOPE)
  endforeach()
endfunction()