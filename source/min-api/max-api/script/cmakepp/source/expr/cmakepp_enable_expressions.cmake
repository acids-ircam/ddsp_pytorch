## `(${CMAKE_CURRENT_LIST_LINE})-><any>`
##
## you need to pass `${CMAKE_CURRENT_LIST_LINE}` for this to work
##
## this macro enables all expressions in the current scope
## it will only work in a CMake file scioe or inside a cmake function scope.
## You CANNOT use it in a loop, if statement, macro etc (everything that has a begin/end)
## Every expression inside that scope (and its subscopes) will be evaluated.  
##
## **Implementation Note**:
## This is achieved by parsing the while cmake file (and thus potentially takes very long)
## Afterwards the line which you pass as an argument is used to find the location of this macro
## every argument for every following expression in the current code scope is scanned for
## `$[...]` brackets which are in turn lexed,parsed and compiled (see `expr()`) and injected
## into the code which is in turn included
macro(cmakepp_enable_expressions line)
  cmakepp_compile_scope_expressions("${line}")
  rethrow(true)
  ans(__cmakepp_enable_expressions_code)
  eval_ref(__cmakepp_enable_expressions_code)
  unset(__cmakepp_enable_expressions_code)
  _return()
endmacro()


