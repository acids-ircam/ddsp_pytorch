

  function(cache_clear cache_key)
    memory_cache_clear("${cache_key}")
    file_cache_clear("${cache_key}")

  endfunction()