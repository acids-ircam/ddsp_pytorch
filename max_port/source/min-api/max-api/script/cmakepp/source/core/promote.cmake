## `(<ref>...)->void`
##
## promotes every specified variable to the PARENT_SCOPE if it is defined
macro(promote )
  foreach(__current_arg ${ARGN})      
    if(DEFINED ${__current_arg})
      set(${__current_arg} "${${__current_arg}}" PARENT_SCOPE)      
    endif()
  endforeach()
endmacro()  