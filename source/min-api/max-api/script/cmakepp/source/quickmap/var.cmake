
## captures a list of variable as a key value pair
function(var)
  foreach(var ${ARGN})
    kv("${var}" "${${var}}")
  endforeach()
endfunction()