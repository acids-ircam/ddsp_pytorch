##
## forces the specified variable reference to become an uri
macro(uri_coerce __uri_ref)
  uri("${${__uri_ref}}")
  ans("${__uri_ref}")
endmacro()