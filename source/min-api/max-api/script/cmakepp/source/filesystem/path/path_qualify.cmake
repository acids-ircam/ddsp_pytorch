
## qualifies the specified variable as a path and sets it accordingly
macro(path_qualify __path_ref)
  path("${${__path_ref}}")
  ans(${__path_ref})
endmacro()
