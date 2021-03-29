## `(<format string...?>-><void>`
##
## formats the specified string and and append it to the template output stream
##
function(template_out_format)
  format("${ARGN}")
  ans(res)
  template_out("${res}")
  return()
endfunction() 