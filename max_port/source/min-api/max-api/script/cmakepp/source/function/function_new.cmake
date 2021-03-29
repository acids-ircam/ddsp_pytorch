# creates a and defines a function (with random name)
function(function_new )
	#generate a unique function id

	set(name_base "${__current_constructor}_${__current_member}")
	string_normalize("${name_base}")
	ans(name_base)

	set(id "${name_base}")
	if("${name_base}" STREQUAL "_")
		set(name_base "__func")
		set(id "__func_1111111111")
	endif()

	while(TRUE)
		if(NOT COMMAND "${id}")
			#declare function
			function("${id}")
				message(FATAL_ERROR "function is declared, not defined")
			endfunction()
			return_ref(id)
		endif()
		#message("making_id because ${id} alreading existers")
		make_guid()
		ans(id)
		set(id "${name_base}_${id}")
	endwhile()


endfunction()

## faster, but less debug info 
macro(function_new )
	identifier(function)
endmacro()