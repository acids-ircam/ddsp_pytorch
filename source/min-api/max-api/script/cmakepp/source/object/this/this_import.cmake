# imports all variables specified as varargs
macro(this_import)
  obj_import("${this}" ${ARGN})
endmacro()