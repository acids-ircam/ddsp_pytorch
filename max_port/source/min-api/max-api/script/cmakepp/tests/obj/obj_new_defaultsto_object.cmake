function(test)
	obj_new()
  ans(obj)
	assert(obj)
	obj_getprototype(${obj} )
	ans(proto)
	assert(proto)
	set(res)
	obj_member_call(${obj} to_string )
	ans(res)
	assert(res)


endfunction()