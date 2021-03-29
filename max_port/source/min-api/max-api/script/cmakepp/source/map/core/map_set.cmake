# set a value in the map
function(map_set this key )
  set(property_ref "${this}.${key}")
  get_property(has_key GLOBAL PROPERTY "${property_ref}" SET)
	if(NOT has_key)
		set_property(GLOBAL APPEND PROPERTY "${this}.__keys__" "${key}")
	endif()
	# set the properties value
	set_property(GLOBAL PROPERTY "${property_ref}" "${ARGN}")
endfunction()
