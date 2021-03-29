
macro(add_dependencies)
  _add_dependencies(${ARGN})
  event_emit(add_dependencies ${ARGN})

endmacro()