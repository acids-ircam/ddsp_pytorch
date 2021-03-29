# compares two maps and returns true if they are equal
# order of list values is important
# order of map keys is not important
# cycles are respected.
function(map_equal lhs rhs)
	# create visited map on first call
	set(visited ${ARGN})
	if(NOT visited)
		map_new()
		ans(visited)
	endif()

	# compare lengths of lhs and rhs return false if they are not equal
	list(LENGTH lhs lhs_length)
	list(LENGTH rhs rhs_length)

	if(NOT "${lhs_length}" EQUAL "${rhs_length}")
		return(false)
	endif()


	# compare each element of list recursively and return result
	if("${lhs_length}" GREATER 1)
		math(EXPR len "${lhs_length} - 1")
		foreach(i RANGE 0 ${len})
			list(GET lhs "${i}" lhs_item)
			list(GET rhs "${i}" rhs_item)
			map_equal("${lhs_item}" "${rhs_item}" ${visited})
			ans(res)
			if(NOT res)
				return(false)
			endif()
		endforeach()
		return(true)
	endif()

	# compare strings values of lhs and rhs and return if they are equal
	if("${lhs}" STREQUAL "${rhs}")
		return(true)
	endif()

	# else lhs and rhs might be maps
	# if they are not return false
	is_map(${lhs})
	ans(lhs_ismap)

	if(NOT lhs_ismap)
		return(false)
	endif()

	is_map(${rhs})	
	ans(rhs_ismap)

	if(NOT rhs_ismap)
		return(false)
	endif()

	# if already visited return true as a parent call will correctly 
	# determine equality
	map_tryget(${visited} ${lhs})
	ans(lhs_isvisited)
	if(lhs_isvisited)
		return(true)
	endif()

	map_tryget(${visited} ${rhs})
	ans(rhs_isvisited)
	if(rhs_isvisited)
		return(true)
	endif()

	# set visited to true
	map_set(${visited} ${lhs} true)
	map_set(${visited} ${rhs} true)

	# compare keys of lhs and rhs	
	map_keys(${lhs} )
	ans(lhs_keys)
	map_keys(${rhs} )
	ans(rhs_keys)

	# order not important
	set_isequal(lhs_keys rhs_keys)
	ans(keys_equal)

	if(NOT keys_equal)		
		return(false)
	endif()

	# compare each property of lhs and rhs recursively
	foreach(key ${lhs_keys})

		map_get(${lhs}  ${key})
		ans(lhs_property_value)
		map_get(${rhs}  ${key})
		ans(rhs_property_value)
		
		map_equal("${lhs_property_value}" "${rhs_property_value}" ${visited})		
		ans(val_equal)
		if(NOT val_equal)
			return(false)
		endif()
	endforeach()

	## everything is equal -> return true
	return(true)
endfunction()

