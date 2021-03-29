
# prints the variables name and value as a STATUS message
macro(print_var varname)
  message(STATUS "${varname}: ${${varname}}")
endmacro()
