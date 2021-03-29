# decodes an encoded empty string
function(string_decode_empty str) 
    string_codes()
  if("${str}" STREQUAL "${empty_code}")
    return("")
  endif()
  return_ref(str)
endfunction()

