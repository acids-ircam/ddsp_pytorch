
function(print_call_counts)
	get_property(props GLOBAL PROPERTY "function_calls")
	set(countfunc "(current) return_truth(\${current} STREQUAL \${it})")
	foreach(prop ${props})
		get_property(call_count GLOBAL PROPERTY "call_count_${prop}")
		get_property(callers GLOBAL PROPERTY "call_count_${prop}_caller")


		message("${prop}: ${call_count}")
	endforeach()
endfunction()