function(proto_declarefunction result)
  string(REGEX MATCH "[a-zA-Z0-9_]+" match "${result}")
  set(function_name "${match}")
  obj_getprototype(${this})
  ans(proto)
	if(NOT proto)
		message(FATAL_ERROR "proto_declarefunction: expected prototype to be present")
	endif()
	set(res ${result})
  set(__current_member ${function_name})
  function_new(${function_name} ${ARGN})
  ans(func)
  obj_set("${proto}" "${function_name}" "${func}")
	#obj_declarefunction(${proto} ${res})
	set(${function_name} "${func}" PARENT_SCOPE)
endfunction()


## shorthand for proto_declarefunction
macro(method result)
  proto_declarefunction("${result}")
endmacro()


# causes the following code inside a constructor to only run once
macro(begin_methods)

endmacro()