## `(<template:<string>>)-><generated content:<string>>`
##
##  this function takes the input string compiles it and evaluates it
##  returning the result of the evaluations
##
function(template_run template)
  template_compile("${template}")
  ans(template_code)
  eval("${template_code}")
  return_ans()
endfunction()
