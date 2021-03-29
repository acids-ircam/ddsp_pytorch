function(test)


	# test to see if the two lists 1 2 3 are equal
	list_equal( 1 2 3  1 2 3)
	ans(res)
	assert(res)


	list_equal( "a;b;c"  "a;b;c")
	ans(res)
	assert(res)

	list_equal( 1 2  1 2 3)
	ans(res)
	assert(NOT res)

	# also works with passing variable value
	set(lst1 1 2 3)
	set(lst2 1 2 3)
	list_equal( ${lst1} ${lst2})
	ans(res)
	assert(res)

	set(lst2 1 2 b )
	list_equal( ${lst1} ${lst2})
	ans(res)
	assert(NOT res)

	# also works with passing list references
	list_equal( lst1 lst2)
	ans(res)
	assert(NOT res)
endfunction()