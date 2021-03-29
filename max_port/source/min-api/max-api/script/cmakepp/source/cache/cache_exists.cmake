

  function(cache_exists cache_key)
    memory_cache_exists("${cache_key}")
    ans(res)
    if(res)
      return_ref(res)
    endif()
    file_cache_exists("${cache_key}")
    ans(res)
    return_ref(res)
  endfunction()
