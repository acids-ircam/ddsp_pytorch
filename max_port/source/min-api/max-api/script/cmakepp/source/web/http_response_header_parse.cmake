

function(http_response_header_parse http_response)
  http_regexes()
  string_encode_semicolon("${http_response}")
  ans(http_response)

  string(REGEX REPLACE "${http_response_header_regex}" "\\1" response_line "${response}")
  string(REGEX REPLACE "${http_response_header_regex}" "\\5" response_headers "${response}")

  string(REGEX REPLACE "${http_response_line_regex}" "\\1" http_version "${response_line}" )
  string(REGEX REPLACE "${http_response_line_regex}" "\\2" http_status_code "${response_line}" )
  string(REGEX REPLACE "${http_response_line_regex}" "\\3" http_reason_phrase "${response_line}" )



  http_headers_parse("${response_headers}")
  ans(http_headers)


  map_new()
  ans(result)
  map_set(${result} "http_version" "${http_version}")
  map_set(${result} "http_status_code" "${http_status_code}")
  map_set(${result} "http_reason_phrase" "${http_reason_phrase}")
  map_set(${result} "http_headers" "${http_headers}")
  return_ref(result)

endfunction()