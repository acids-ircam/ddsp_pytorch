function(map_set_hidden map property)
  set_property(GLOBAL PROPERTY "${map}.${property}" ${ARGN})
endfunction()