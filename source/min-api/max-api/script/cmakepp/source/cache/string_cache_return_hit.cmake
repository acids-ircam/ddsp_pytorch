
macro(string_cache_return_hit cache_location key)
  string_cache_hit("${cache_location}" "${key}")
  ans(hit)
  if( hit)
    string_cache_get("${cache_location}" "${key}")
    return_ans()
  endif()
endmacro()  
