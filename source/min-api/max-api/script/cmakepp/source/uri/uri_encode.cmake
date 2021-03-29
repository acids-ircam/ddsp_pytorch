## encodes a string to uri format 
## if you can pass decimal character codes  which are encoded 
## if you do not pass any codes  the characters  recommended by rfc2396
## are encoded
function(uri_encode str ) 

  if(NOT ARGN)
    uri_recommended_to_escape()
    ans(codes)
    list(APPEND codes)
  else()
    set(codes ${ARGN})
  endif()

  foreach(code ${codes})
    string(ASCII "${code}" char)
    dec2hex("${code}")
    ans(hex)
    # pad with zero
    if("${code}" LESS  16)
      set(hex "0${hex}")
    endif()

    string(REPLACE "${char}" "%${hex}" str "${str}" )
  endforeach()

  return_ref(str)
endfunction()

