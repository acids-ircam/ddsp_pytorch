
function(string_cache_location cache_location key)
  cmakepp_config(cache_dir)
  ans(cache_dir)
  path_qualify_from("${cache_dir}" "${cache_location}/${key}")
  ans(location)
  return_ref(location)
endfunction()