
function(target_copy_imports target)
message(FATAL_ERROR "not implemented")
  target_get("${target}" IMPORTED)
  ans(isImported)
  if(NOT isImported)
    message(WARNING "${target} is not IMPORTED - cannot copy to output directory")
    return()
  endif()

  target_get("${target}" TYPE)
  ans(type)


  if(NOT "${type}" STREQUAL "SHARED_LIBRARY")
    message(WARNING "${target} is not a shared library")
  endif()

  target_get("${target}" IMPORTED_LOCATION)
  ans(sharedLibPath)
  target_get("${target}" RUNTIME_OUTPUT_DIRECTORY)
  ans(outputPath)

  message("cp ${sharedLibPath} to ${outputPath}")

endfunction()