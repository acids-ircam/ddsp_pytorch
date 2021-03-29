function(test)
	
	#call_function("function(fu arg1 arg2)  \n set(result \"\${arg1} \${arg1} \${arg2} \${arg2}\" PARENT_SCOPE) \n endfunction()" a b)
	function_import(" function(fu arg1 arg2)  \n set(result \"\${arg1} \${arg1} \${arg2} \${arg2}\" PARENT_SCOPE) \n endfunction()" as function_to_call REDEFINE )
function_to_call(a b)
	assert("${result}" STREQUAL "a a b b" )

endfunction()