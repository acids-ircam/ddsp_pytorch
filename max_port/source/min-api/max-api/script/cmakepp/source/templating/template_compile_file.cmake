##
## `(<file path>)-> <cmake code>`
## 
## reads the contents of the specified path and generates a template from it
## * return
##   * the generated template code
##
function(template_compile_file path)
  fread("${path}")
  ans(content)
  template_compile("${content}")
  return_ans()
endfunction()