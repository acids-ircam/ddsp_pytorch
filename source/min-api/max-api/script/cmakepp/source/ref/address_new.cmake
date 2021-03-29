function(address_new)
	address_set(":0" 0)	
	function(address_new)
		address_get(":0" )
		ans(index)
		math(EXPR index "${index} + 1")
		address_set(":0" "${index}")
		if(ARGN)
		#	set(type "${ARGV0}")
			address_set(":${index}.__type__" "${ARGV0}")
		endif()
		return(":${index}")
	endfunction()

	address_new(${ARGN})
	return_ans()
endfunction()

## optimized version
function(address_new)
	set_property(GLOBAL PROPERTY ":0" 0 )
	function(address_new)
		get_property(index GLOBAL PROPERTY ":0")
		math(EXPR index "${index} + 1")
		set_property(GLOBAL PROPERTY ":0" ${index} )
		if(ARGN)
			set_property(GLOBAL PROPERTY ":${index}.__type__" "${ARGV0}")
		endif()
		set(__ans ":${index}" PARENT_SCOPE)
	endfunction()

	address_new(${ARGN})
	return_ans()
endfunction()

## faster
macro(address_new)
	set_property(GLOBAL PROPERTY ":0" 0)
	macro(address_new)
		get_property(__ans GLOBAL PROPERTY ":0")
		math(EXPR __ans "${__ans} + 1")
		set_property(GLOBAL PROPERTY ":0" "${__ans}")
		set_property(GLOBAL PROPERTY ":${__ans}")
		set(__ans ":${__ans}")
	endmacro()
	address_new(${ARGN})
endmacro()