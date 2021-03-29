
function(map_keys_sort map)
  get_property(keys GLOBAL PROPERTY "${map}.__keys__")
  if(keys)
    list(SORT keys)
    set_property(GLOBAL PROPERTY "${map}.__keys__" ${keys})
  endif()
endfunction()
