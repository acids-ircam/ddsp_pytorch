##
## `()-><template output:<address>>`
##
## fails if not executed inside of a template else returns the 
## template output ref
##
function(template_guard)
  template_output_stream()
  ans(ref)
  if(NOT ref)
    message(FATAL_ERROR "call may only occure inside of a template")
  endif()  
  return(${ref})
endfunction()

