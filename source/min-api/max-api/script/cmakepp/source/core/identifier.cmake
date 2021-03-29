
## 
##
## returns an identifier of the form `__{ARGN}_{unique}`
## the idetnfier will not be a defined function 
## nor a defined variable, nor a existing global 
## property.  it is unique to the execution of cmake
## and can be used as a function name
function(identifier)
  #string_codes()
  while(true)
    make_guid()
    ans(guid)
    set(identifier "__${ARGN}_${guid}")
    if(NOT COMMAND "${identifier}" AND NOT "${identifier}")
      return_ref(identifier)
    endif()
  endwhile()
  message(FATAL_ERROR "code never reached")
endfunction()