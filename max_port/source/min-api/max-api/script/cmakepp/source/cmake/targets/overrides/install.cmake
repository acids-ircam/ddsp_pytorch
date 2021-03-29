# overwrites install command
#  emits event install and on_target_added(install ${ARGN)
# registers install target globally
macro(install)
  _install(${ARGN})
  event_emit(install ${ARGN})

  event_emit(on_target_added install install ${ARGN})
  target_register(install install ${ARGN})

endmacro()