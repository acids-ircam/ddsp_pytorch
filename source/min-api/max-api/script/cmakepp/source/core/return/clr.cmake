# used to clear the __ans variable. may also called inside a function with argument PARENT_SCOPE to clear
# parent __ans variable
macro(clr)
  set(__ans ${ARGN})
endmacro()