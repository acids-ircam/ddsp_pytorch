function(test)
	function_string_get( "function(fu)\nmessage(\${ARGN} \${arg})\nendfunction()")
  ans(str)

	assert(str)
	assert(str STREQUAL "function(fu)\nmessage(\${ARGN} \${arg})\nendfunction()")
	#message("${str}")
endfunction()