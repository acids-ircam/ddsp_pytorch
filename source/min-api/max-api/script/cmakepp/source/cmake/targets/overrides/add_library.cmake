# overwrites add_library
# same function as cmakes original add_library
# emits the event add_library with all parameters of the add_library call
# emits the event on_target_added library with all parameters of the call added
# registers the target globally so it can be iterated via 
macro(add_library)
  _add_library(${ARGN})
  event_emit(add_library ${ARGN})

  event_emit(on_target_added library ${ARGN})
  target_register(${ARGN})


  
endmacro()
