## decodes an uri encoded string ie replacing codes %XX with their ascii values
 function(uri_decode str)
  set(hex "[0-9A-Fa-f]")
  set(encoded "%(${hex}${hex})")
  string(REGEX MATCHALL "${encoded}" matches "${str}")

  list(REMOVE_DUPLICATES matches)
  foreach(match ${matches})
    string(SUBSTRING "${match}" 1 -1  hex_code)
    hex2dec("${hex_code}")
    ans(dec_code)
    string(ASCII "${dec_code}" char)
    string(REPLACE "${match}" "${char}" str "${str}")
  endforeach()
  return_ref(str)

 endfunction()