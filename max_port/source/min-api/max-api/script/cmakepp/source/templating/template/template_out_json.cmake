## `(<structured data...>) -> <void>`
## 
## writes the serialized data to the templates output
## fails if not called inside a template
##
function(template_out_json)
  json_indented(${ARGN})
  ans(res)
  template_out("${res}")
  return()
endfunction()
