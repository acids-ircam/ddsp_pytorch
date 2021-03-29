
macro(memory_cache_return_hit cache_key)
  memory_cache_get("${cache_key}")
  ans(__cache_return)
  if(__cache_return)
    return_ref(__cache_return)
  endif()
endmacro()