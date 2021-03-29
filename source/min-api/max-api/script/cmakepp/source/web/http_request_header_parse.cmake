


function(http_request_header_parse http_request)
  http_regexes()

  string_encode_semicolon("${http_request}")
  ans(http_request)

  string(REGEX REPLACE "${http_request_header_regex}" "\\1" http_request_line "${http_request}")
  string(REGEX REPLACE "${http_request_header_regex}" "\\5" http_request_headers "${http_request}")

  string(REGEX REPLACE "${http_request_line_regex}" "\\1" http_method "${http_request_line}")
  string(REGEX REPLACE "${http_request_line_regex}" "\\2" http_request_uri "${http_request_line}")
  string(REGEX REPLACE "${http_request_line_regex}" "\\3" http_version "${http_request_line}")


  
  http_headers_parse("${http_request_headers}")
  ans(http_headers)

  map_new()
  ans(result)

  map_set(${result} http_method "${http_method}")
  map_set(${result} http_request_uri "${http_request_uri}")
  map_set(${result} http_version "${http_version}")
  map_set(${result} http_headers ${http_headers})

  return_ref(result)
endfunction()