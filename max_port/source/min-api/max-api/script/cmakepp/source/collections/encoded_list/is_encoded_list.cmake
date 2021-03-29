##
##
## returns true iff the arguments passed are in encoded list format
function(is_encoded_list)
  if("${ARGN}" MATCHES "[]")
    set(__ans true PARENT_SCOPE)
  else()
    set(__ans false PARENT_SCOPE)
  endif()
endfunction()