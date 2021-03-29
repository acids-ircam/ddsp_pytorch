# writs argn to the speicified file creating it if it does not exist and 
# overwriting it if it does.
function(fwrite path)
  path_qualify(path)
  file(WRITE "${path}" "${ARGN}")
  event_emit(on_fwrite "${path}")
  return_ref(path)
endfunction()