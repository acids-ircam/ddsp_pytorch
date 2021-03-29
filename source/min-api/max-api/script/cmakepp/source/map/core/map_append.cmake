# appends a value to the end of a map entry
function(map_append map key)
  get_property(isset GLOBAL PROPERTY "${map}.${key}" SET)
	if(NOT isset)
		map_set(${map} ${key} ${ARGN})
		return()
	endif()
  set_property(GLOBAL APPEND PROPERTY "${map}.${key}" ${ARGN})
endfunction()
