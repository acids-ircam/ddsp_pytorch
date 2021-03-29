macro(include_directories)
  _include_directories(${ARGN})
  event_emit(include_directories "${ARGN}")
endmacro()