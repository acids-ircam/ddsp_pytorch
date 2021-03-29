
# injects code into  function (right after function is called) and returns result
function(function_string_rename input_function new_name) 
	function_string_get( "${input_function}")
	ans(function_string)
	function_signature_regex(regex)

	function_lines_get( "${input_function}")
	ans(lines)
	
	foreach(line ${lines})
		string(REGEX MATCH "${regex}" found "${line}")
		if(found)
			string(REGEX REPLACE "${regex}"  "\\1(${new_name} \\3)" new_line "${line}")
			string_replace_first("${input_function}" "${line}" "${new_line}")
			ans(input_function)
			break()
		endif()
	endforeach()
	return_ref(input_function)
endfunction()