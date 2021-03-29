
## registers a parameter definition for the specified function name
## the function may parse the input values from this definition
## help function can parse the input parameters
function(parameter_definition name)
  if(name)
    ## cache values until cmakepp is loaded
    set_property(GLOBAL APPEND PROPERTY __param_defs ${name})
    set_property(GLOBAL PROPERTY "__param_defs.${name}" "${ARGN}")
    if(NOT cmakepp_is_loaded)
      return()
    endif()

    ## actual function
    function(parameter_definition name)
      typed_value_definitions("${name}" ${ARGN})
      ans(definitions)
      map_set(__global_definitions "${name}" "${definitions}")
      return(${definitions})
    endfunction()
  endif()
  if(NOT cmakepp_is_loaded)
    return()
  endif()

  ## load cached values
  get_property(names GLOBAL PROPERTY __param_defs)
  foreach(name ${names})
    get_property(defstring GLOBAL PROPERTY "__param_defs.${name}")
    parameter_definition("${name}" ${defstring})
  endforeach()


endfunction()