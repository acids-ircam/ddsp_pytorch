
function(memory_cache_clear cache_key)
  memory_cache_key("${cache_key}")
  ans(key)
  map_set_hidden(memory_cache_entries "${key}")
  return()
endfunction()