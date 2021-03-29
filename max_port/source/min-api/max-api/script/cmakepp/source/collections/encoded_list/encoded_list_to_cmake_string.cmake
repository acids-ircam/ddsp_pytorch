
function(encoded_list_to_cmake_string)
  ## free token 
  set(result "${ARGN}")
  string(REPLACE ";" "" result "${result}")
  cmake_string_escape2("${result}")
  ans(result)
  string(REPLACE "" " " result "${result}")
  encoded_list_decode("${result}")
  ans(result)
  return_ref(result)
endfunction()