
macro(target_compile_options)
  _target_compile_options(${ARGN})
  event_emit(target_compile_options ${ARGN})

endmacro()