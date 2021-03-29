

function(file_cache_update cache_key)
  file_cache_key("${cache_key}")
  ans(path)
  qm_serialize("${ARGN}")
  ans(ser)
  file(WRITE "${path}" "${ser}")
  return()  
endfunction()
