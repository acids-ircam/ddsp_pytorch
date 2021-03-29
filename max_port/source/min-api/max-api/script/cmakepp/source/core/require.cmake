function(require file)
  file(GLOB_RECURSE res "${file}")

  if(NOT res)
    message(FATAL_ERROR "could not find required file for '${file}'")
  endif()

  foreach(file ${res})
    include("${file}")
  endforeach()

endfunction()
