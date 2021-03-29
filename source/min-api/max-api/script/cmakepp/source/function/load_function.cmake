# reads a functions and returns it
function(load_function result file_name)	
	file(READ ${file_name} func)	
	set(${result} ${func} PARENT_SCOPE)
endfunction()