
macro(add_test)
  _add_test(${ARGN})
  event_emit(add_test ${ARGN})
  event_emit(on_target_added test ${ARGN})
  target_register(${ARGN})

endmacro()