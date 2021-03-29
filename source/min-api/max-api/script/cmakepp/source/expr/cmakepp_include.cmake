## `()->`
##
## includes the specified cmakepp file (compiling it)
macro(cmakepp_include path)
  cmakepp_compile_file("${path}")
  include("${__ans}")
endmacro()