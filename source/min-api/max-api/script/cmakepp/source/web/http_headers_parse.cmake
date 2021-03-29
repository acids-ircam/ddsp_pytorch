

function(http_headers_parse http_headers)
  http_regexes()
  string_encode_semicolon("${http_headers}")
  ans(http_headers)

  string(REGEX MATCHALL "${http_header_regex}" http_header_lines "${http_headers}")

  map_new()
  ans(result)
  foreach(header_line ${http_header_lines})
    string(REGEX REPLACE "${http_header_regex}" "\\1" header_key "${header_line}")
    string(REGEX REPLACE "${http_header_regex}" "\\2" header_value "${header_line}")
    string_decode_semicolon("${header_value}")
    ans(header_value)
    map_set(${result} "${header_key}" "${header_value}")
  endforeach()

  return_ref(result)
endfunction()
