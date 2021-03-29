## sets a target config property
## ie.  target_config_set(mytarget RELEASE IMPORTED_LOCATION value)
## translates to set_property(TARGET mytarget PROPERTY IMPORTED_LOCATION_RELEASE value )
function(target_config_set targetName targetConfig propertyName)
    if(NOT "${targetConfig}_" STREQUAL "_")
        string_toupper("${targetConfig}")
        ans(targetConfig)


        list_contains(CMAKE_CONFIGURATION_TYPES ${targetConfig})
        ans(isvalid)
        if(NOT isvalid)
            message(FATAL_ERROR "cannot set target property '${propertyName}' for config '${targetConfig}' because this type of configuration does not exist  ")
            return(false)
        endif()

        set(propertyName "${propertyName}_${targetConfig}")
    endif()
    target_set("${targetName}" "${propertyName}" ${ARGN})
    return(true)
endfunction()