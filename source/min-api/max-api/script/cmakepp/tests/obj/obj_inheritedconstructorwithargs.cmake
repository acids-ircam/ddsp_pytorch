function(test)

	function(ConstructorWithArgsTestBase a)
		this_set(arg1 "base_${a}")
	endfunction()
	function(ConstructorWithArgsTestDerived arg1 arg2)
		this_inherit(ConstructorWithArgsTestBase ${arg1})
		this_set(arg2 "derived_${arg2}")
	endfunction()


	obj_new( ConstructorWithArgsTestDerived a b)
ans(uut)
	map_navigate(res "uut.arg1")
	assert("${res}" STREQUAL "base_a")
	map_navigate(res "uut.arg2")
	assert("${res}" STREQUAL "derived_b")


endfunction()