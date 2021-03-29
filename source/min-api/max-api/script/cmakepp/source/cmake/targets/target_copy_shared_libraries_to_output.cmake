## 
## adds a custom command to executableTarget which copies 
## the shared libraries of its dependencies to the target's directory
##
function(target_copy_shared_libraries_to_output executableTarget)
  target_get("${executableTarget}" TYPE)
  ans(exeType)
  if(NOT "${exeType}" STREQUAL "EXECUTABLE")
    message(WARNING "target_copy_shared_libraries_to_output only works for executable")
    return()
  endif()
  target_get(${executableTarget} LINK_LIBRARIES)
  ans(dependencies)
  foreach(dependency ${dependencies})

    target_get("${dependency}" TYPE)
    ans(targetType)
    if(NOT "${targetType}" STREQUAL "SHARED_LIBRARY")
      continue()
    endif()

    target_get(${dependency} IMPORTED)
    ans(isImported)
    if(isImported)
      add_custom_command(TARGET "${executableTarget}"  POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E  copy_if_different $<TARGET_PROPERTY:${dependency},LOCATION_$<CONFIG>>  "$<TARGET_FILE_DIR:${executableTarget}>"
        COMMENT "copying dlls for '${dependency}' to output directory of '${executableTarget}' ..."
      )
    else()
      add_custom_command(TARGET "${executableTarget}"  POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E  copy_if_different $<TARGET_FILE:${dependency}>  "$<TARGET_FILE_DIR:${executableTarget}>"
        COMMENT "copying dlls for '${dependency}' to output directory of '${executableTarget}' ..."
      )
    endif()
  endforeach()

endfunction()