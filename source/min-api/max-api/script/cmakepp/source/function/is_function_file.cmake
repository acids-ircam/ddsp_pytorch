function(is_function_file result function_file)
	path("${function_file}")
	ans(function_file)
	
	if(NOT EXISTS "${function_file}")
		return_value(false)
	endif()

	if(IS_DIRECTORY "${function_file}")
		return_value(false)
	endif()

	file(READ "${function_file}" input)
	if(NOT input)
		return_value(false)
	endif()
	#is_function_string(res ${input})
	is_function(res "${input}")
	
	return_value(${res})
endfunction()