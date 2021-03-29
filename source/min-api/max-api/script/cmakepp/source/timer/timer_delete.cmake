## removes the specified timer
function(timer_delete id)
  map_remove(__timers "${id}")
  return()
endfunction()
