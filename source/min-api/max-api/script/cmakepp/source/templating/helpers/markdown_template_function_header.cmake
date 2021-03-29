
function(markdown_template_function_header signature)
  assign(function_name = function_def.function_args[0])

  return("[**`${function_name}${signature}`**](${template_path})")
endfunction()
