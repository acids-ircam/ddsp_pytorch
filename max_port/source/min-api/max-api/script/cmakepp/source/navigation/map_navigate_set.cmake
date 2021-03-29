function(map_navigate_set navigation_expression)
	cmake_parse_arguments("" "FORMAT" "" "" ${ARGN})
	set(args)
	if(_FORMAT)
		foreach(arg ${_UNPARSED_ARGUMENTS})
			format( "${arg}")
			ans(formatted_arg)
			list(APPEND args "${formatted_arg}")
		endforeach()
	else()
		set(args ${_UNPARSED_ARGUMENTS})
	endif()
	# path is empty => ""
	if(navigation_expression STREQUAL "")
		return_value("")
	endif()

	# split off reference from navigation expression
	unset(ref)
	string(REGEX MATCH "^[^\\[|\\.]*" ref "${navigation_expression}")
	string(LENGTH "${ref}" len )
	string(SUBSTRING "${navigation_expression}" ${len} -1 navigation_expression)

	# rest of navigation expression is empty, first is a var
	if(NOT navigation_expression)

		set(${ref} "${args}" PARENT_SCOPE)
		return()
	endif()
	



	# match all navigation expression parts
	string(REGEX MATCHALL  "(\\[([0-9][0-9]*)\\])|(\\.[a-zA-Z0-9_\\-][a-zA-Z0-9_\\-]*)" parts "${navigation_expression}")
	
	# loop through parts and try to navigate 
	# if any part of the path is invalid return ""

	set(current "${${ref}}")
	
	
	while(parts)
		list(GET parts 0 part)
		list(REMOVE_AT parts 0)
		
		string(REGEX MATCH "[a-zA-Z0-9_\\-][a-zA-Z0-9_\\-]*" index "${part}")
		string(SUBSTRING "${part}" 0 1 index_type)	



		#message("current ${current}, parts: ${parts}, current_part: ${part}, current_index ${index} current_type : ${index_type}")
		# first one could not be ref so create ref and set output
		is_address("${current}")
		ans(isref)
		
		if(NOT isref)
			map_new()
    	ans(current)
			set(${ref} ${current} PARENT_SCOPE)
		endif()		
		
		# end of navigation string reached, set value
		if(NOT parts)
			map_set(${current} ${index} "${args}")
			return()
		endif()

		
		map_tryget(${current}  "${index}")
		ans(next)
		# create next element in change
		if(NOT next)
			map_new()
    	ans(next)
			map_set(${current} ${index} ${next})
		endif()

		# if no next element exists its an error
		if(NOT next)
			message(FATAL_ERROR "map_navigate_set: path is invalid")
		endif()

		set(current ${next})

		
	endwhile()
endfunction()