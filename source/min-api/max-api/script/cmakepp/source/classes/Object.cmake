function(Object)
	#formats the current object 
	proto_declarefunction(to_string)

	function(${to_string} )
		set(res)
#		debug_message("to_string object ${this}")
		obj_keys(${this} keys)

		foreach(key ${keys})
			obj_get(${this}  ${key})				
			ans(value)
			map_has(${this}  ${key})
			ans(is_own)	
			if(value)
				is_function(function_found ${value})
				is_object(object_found ${value})
			endif()
			
			
			if(function_found)
				set(value "[function]")
			elseif(object_found)
				get_filename_component(fn ${value} NAME_WE)
				obj_gettype(${value} )
				ans(type)
				if(NOT type)
					set(type "")
				endif()
				set(value "[object ${type}:${fn}]")
			else()
				set(value "\"${value}\"")
			endif()
			if(is_own)
				set(is_own "*")
			else()
				set(is_own " ")
			endif()

			set(nextValue "${is_own}${key}: ${value}")

			if(res)
				set(res "${res}\n ${nextValue}, ")	
			else()
				set(res " ${nextValue}, ")
			endif()
		endforeach()

		set(res "{\n${res}\n}")
		return_ref(res)
	endfunction()

	# prints the current object to the console
	proto_declarefunction(print)
	function(${print})
		#debug_message("printing object ${this}")
		obj_member_call(${this} "to_string" str )
		message("${str}")
	endfunction()
endfunction()


