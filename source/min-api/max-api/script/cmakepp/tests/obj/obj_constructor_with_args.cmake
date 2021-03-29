function(test)

	function(ConstructorWithArgsTest arg1 arg2)
		this_set(arg1 ${arg1})
		this_set(arg2 ${arg2})
	endfunction()

	obj_new( ConstructorWithArgsTest a b)
ans(uut)
	map_navigate(res "uut.arg1")
	assert("${res}" STREQUAL "a")
	map_navigate(res "uut.arg2")
	assert("${res}" STREQUAL "b")


endfunction()