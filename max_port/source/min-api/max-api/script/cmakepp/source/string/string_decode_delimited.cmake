## tries to parse a delimited string
## returns either the original or the parsed delimited string
## delimiters can be specified via varargs
## see also string_take_delimited
function(string_decode_delimited str)
  string_take_delimited(str ${ARGN})
  ans(res)
  if("${res}_" STREQUAL "_")
    return_ref(str)
  endif()
  return_ref(res)
endfunction()