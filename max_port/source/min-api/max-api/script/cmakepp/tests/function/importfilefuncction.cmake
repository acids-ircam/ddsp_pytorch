function(test)
	set(funcfile "${test_dir}/importfiletest.cmake")
	file(REMOVE "${test_dir}/tests")
	file(WRITE ${funcfile} "function(myfu123unique) \n endfunction()")

	assert(NOT COMMAND thistestfunction)
	assert(NOT COMMAND myfu123unique)
	function_import(${funcfile} as thistestfunction)
	assert(COMMAND thistestfunction)
	assert(NOT COMMAND myfu123unique)
	
endfunction()