## 
##
## when not to use: if your data degrades when evaluated by a macro
## for example escapes are resolved
macro(return)
  set(__ans "${ARGN}" PARENT_SCOPE)
	_return()
endmacro()