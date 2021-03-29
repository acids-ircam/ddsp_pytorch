

function(file_cache_exists cache_key)
  file_cache_key("${cache_key}")
  ans(path)
  if(EXISTS "${path}")
    return(true)
  endif()
  return(false)
endfunction()