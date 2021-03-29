

function(file_cache_get cache_key)
  file_cache_key("${cache_key}")
  ans(path)
  if(EXISTS "${path}")
    qm_deserialize_file("${path}")
    return_ans()
  endif()
  return()
endfunction()