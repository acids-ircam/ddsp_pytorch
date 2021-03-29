## `(<list&> <element:<any...>>)-><bool>`
##
## returns true if list contains every element specified 
##
function(list_contains __list_contains_lst)
	foreach(arg ${ARGN})
		list(FIND ${__list_contains_lst} "${arg}" idx)
		if(${idx} LESS 0)
			return(false)
		endif()
	endforeach()
	return(true)
endfunction()