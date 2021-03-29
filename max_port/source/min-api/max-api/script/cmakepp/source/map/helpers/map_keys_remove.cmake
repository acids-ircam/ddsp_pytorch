
function(map_keys_remove map)
  get_property(keys GLOBAL PROPERTY "${map}.__keys__" )
  if(keys AND ARGN)
    list(REMOVE_ITEM keys ${ARGN})
    set_property(GLOBAL PROPERTY "${map}.__keys__" ${keys})
  endif()
endfunction()