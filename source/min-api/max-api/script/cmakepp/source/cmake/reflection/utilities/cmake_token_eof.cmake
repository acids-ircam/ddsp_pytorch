  
function(cmake_token_eof)
  map_new()
  ans(token)
  map_set(${token} type eof)
  map_set(${token} value "")
  map_set(${token} literal_value "")
  return_ref(token)
endfunction()