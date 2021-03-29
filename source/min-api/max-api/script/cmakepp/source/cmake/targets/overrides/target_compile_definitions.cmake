
macro(target_compile_definitions)
  _target_compile_definitions(${ARGN})
  event_emit(target_compile_definitions ${ARGN})

endmacro()