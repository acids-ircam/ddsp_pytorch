## `()-><generated content:<string>>`
## ends the current template and returns the generated content
function(template_end)
  template_guard()
  address_get("${__template_output_stream}")
  return_ans()
endfunction()
