

function(file_cache_clear cache_key)
  file_cache_key("${cache_key}")
  ans(path)
  if(EXISTS "${path}")
    file(REMOVE "${path}")
  endif()
  return()
endfunction()