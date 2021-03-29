## `(<map>)-><uint>`
##
## returns the number of elements for the specified map
macro(map_count map)
  map_keys("${map}")
  list(LENGTH __ans __ans)
endmacro()