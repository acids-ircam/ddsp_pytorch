
macro(add_executable)
  _add_executable(${ARGN})
  event_emit(add_executable ${ARGN})
  event_emit(on_target_added executable ${ARGN})
  target_register(${ARGN})
endmacro()