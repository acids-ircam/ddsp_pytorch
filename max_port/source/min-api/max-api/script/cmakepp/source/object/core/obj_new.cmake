function(obj_new)
	set(args ${ARGN})
	list_pop_front( args)
	ans(constructor)
	list(LENGTH constructor has_constructor)
	if(NOT has_constructor)
		set(constructor Object)
	endif()
	

	if(NOT COMMAND "${constructor}")	
		message(FATAL_ERROR "obj_new: invalid type defined: ${constructor}, expected a cmake function")
	endif()

	obj_type_get(${constructor})
	ans(base)

	map_new()
	ans(instance)

	obj_setprototype(${instance} ${base})


	set(__current_constructor ${constructor})
	obj_member_call(${instance} __constructor__ ${args})
	ans(res)


	if(res)
		set(instance "${res}")
	endif()

	map_set_special(${instance} "object" true)

	return_ref(instance)
endfunction()