# ensures that the specified vars are a map
# parsing structured data if necessary
  macro(map_ensure)
    foreach(__map_ensure_arg ${ARGN})
      obj("${${__map_ensure_arg}}")
      ans("${__map_ensure_arg}")
    endforeach()
  endmacro()