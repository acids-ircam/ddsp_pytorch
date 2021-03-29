 ## `()-><bool>`
##
## sets the value of the specified prop if it does not exist
## ie if map_has returns false for the specified property
## returns true iff value was set
function(map_set_default map prop)
  map_has("${map}" "${prop}")
  if(__ans)
    return(false)
  endif()
  map_set("${map}" "${prop}" ${ARGN})
  return(true)
endfunction()
