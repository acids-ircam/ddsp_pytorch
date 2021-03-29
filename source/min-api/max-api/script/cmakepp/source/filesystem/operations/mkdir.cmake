# creates a new directory
function(mkdir path)    
  path("${path}")
  ans(path)
  file(MAKE_DIRECTORY "${path}")
  event_emit(on_mkdir "${path}")
  return_ref(path)
endfunction()

