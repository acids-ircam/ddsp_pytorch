
# pushes the specified vars to the parent scope
macro(vars_elevate)
  set(args ${ARGN})
  foreach(arg ${args})
    set("${arg}" ${${arg}} PARENT_SCOPE)
  endforeach()
endmacro()
