# overwrites target_link_libraries
# emits the event target_link_libraries
macro(target_link_libraries)
  _target_link_libraries(${ARGN})
  target_link_libraries_register(${ARGN})
  event_emit(target_link_libraries ${ARGN})
  
endmacro()

function(target_link_libraries_register target)
  
endfunction()