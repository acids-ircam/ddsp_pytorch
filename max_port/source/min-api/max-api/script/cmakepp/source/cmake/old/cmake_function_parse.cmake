
function(cmake_function_parse code)
  cmake_function_signature("${code}")
  ans(signature)
  map_set(${signature} code "${code}")
  return_ref()
endfunction()
