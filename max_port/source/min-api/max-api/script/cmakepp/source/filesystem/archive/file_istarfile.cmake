## returns true if the specified file is a tar archive 
function(file_istarfile file)
	path_qualify(file)
	if(NOT EXISTS "${file}")
		return(false)
	endif()
	if(IS_DIRECTORY "${file}")
		return(false)
	endif()
	tar_lean(ztvf "${file}")
	ans_extract(res)
	ans(rest)

	if(res)
		return(false)
	endif()

	return(true)
	
endfunction()




