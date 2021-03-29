
  function(cache_update cache_key value)
    memory_cache_update("${cache_key}" "${value}" ${ARGN})
    file_cache_update("${cache_key}" "${value}" ${ARGN})
  endfunction()