
## returns the values of multiple refs
macro(return_refs)
  set(__ans)
  foreach(arg ${ARGN})
    list(APPEND __ans "${${arg}}")
  endforeach()
  set(__ans "${__ans}" PARENT_SCOPE)
  _return()
endmacro()