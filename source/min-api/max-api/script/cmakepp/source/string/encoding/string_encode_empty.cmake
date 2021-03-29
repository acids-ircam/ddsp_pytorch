# encodes an empty element
function(string_encode_empty str)
  message("huh")
  string_codes()

  if("_${str}" STREQUAL "_")
    return("${empty_code}")
  endif()
  return_ref(str)
endfunction()




