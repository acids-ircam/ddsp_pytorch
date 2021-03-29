## returns the list of known timers
function(timers_get)
  map_keys(__timers)
  ans(timers)
  return_ref(timers)
endfunction()
