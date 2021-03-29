## captures a new map from the given variables
## example
## set(a 1)
## set(b 2)
## set(c 3)
## map_capture_new(a b c)
## ans(res)
## json_print(${res})
## --> 
## {
##   "a":1,
##   "b":2,
##   "c":3 
## }
function(map_capture_new)
  map_new()
  ans(__map_capture_new_map)
  map_capture(${__map_capture_new_map} ${ARGN})
  return(${__map_capture_new_map})
endfunction()