function(is_address ref)
  list(LENGTH ref len)
  if(NOT ${len} EQUAL 1)
    return(false)
  endif()
	string(REGEX MATCH "^:" res "${ref}" )
	if(res)
		return(true)
	endif()
	return(false)
endfunction()

## faster - does not work in all cases
macro(is_address ref)
  if("_${ref}" MATCHES "^_:[^;]+$")
    set(__ans true)
  else()  
    set(__ans false)
  endif()
endmacro()


## correcter
## the version above cannot be used because 
## is_address gets arbirtray data - and since macros evaluate 
## arguments a invalid ref could be ssen as valid 
## or especially \\ fails because it beomes \ and causes an error
function(is_address ref)
  if("_${ref}" MATCHES "^_:[^;]+$")
    set(__ans true PARENT_SCOPE)
  else()  
    set(__ans false PARENT_SCOPE)
  endif()
endfunction()