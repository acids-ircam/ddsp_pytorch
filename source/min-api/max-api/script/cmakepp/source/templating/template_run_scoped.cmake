
function(template_run_scoped scope)
  map_import_properties_all("${scope}")
  template_run("${ARGN}")
  return_ans()
endfunction()