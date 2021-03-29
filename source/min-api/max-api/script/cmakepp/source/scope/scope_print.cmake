# print the local scope as json
function(scope_print)
  scope_export_map()
  ans(scope)
  json_print(${scope})
  return()
endfunction()