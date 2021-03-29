
#returns a value 
# expects a variable called result to exist in function signature
# may only be used inside functions
macro(return_value)
  if(NOT result)
    message(FATAL_ERROR "expected a variable called result to exist in function")
    return()
  endif()
  set(${result} ${ARGN} PARENT_SCOPE)
  return(${ARGN})
endmacro()