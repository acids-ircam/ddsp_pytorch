

function(save_function file_name function_string)
	
	file(WRITE "${file_name}" "${function_string}")
endfunction()