function(address_append ref)
	set_property( GLOBAL APPEND PROPERTY "${ref}" "${ARGN}")
endfunction()