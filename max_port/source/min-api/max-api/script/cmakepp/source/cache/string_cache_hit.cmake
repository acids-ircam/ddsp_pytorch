
function(string_cache_hit cache_location key)
  string_cache_location("${cache_location}" "${key}")
  ans(location)
  return_truth(EXISTS "${location}")
endfunction()