macro(add_custom_target)
  _add_custom_target(${ARGN})


  event_emit(add_custom_target ${ARGN})
  event_emit(on_target_added custom ${ARGN})
  target_register(${ARGN})
endmacro()