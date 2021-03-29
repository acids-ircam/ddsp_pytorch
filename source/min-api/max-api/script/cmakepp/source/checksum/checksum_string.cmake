## `(<string> [--algorithm <hash algorithm> = MD5])-><checksum>`
## `<hash algorithm> ::= "MD5"|"SHA1"|"SHA224"|"SHA256"|"SHA384"|"SHA512"`
##
## this function takes any string and computes the hash value of it using the 
## hash algorithm specified (which defaults to  MD5)
## returns the checksum
## 
function(checksum_string str)
  set(args ${ARGN})
  list_extract_labelled_value(args --algorithm)
  ans(algorithm)
  if(NOT algorithm)
    set(algorithm MD5)
  endif()
  string("${algorithm}"  checksum "${str}" )
  return_ref(checksum)
endfunction()