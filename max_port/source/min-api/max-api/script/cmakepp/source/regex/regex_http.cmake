
macro(http_regexes)
  #https://www.ietf.org/rfc/rfc2616
  set(http_version_regex "HTTP/[0-9]\\.[0-9]")
  set(http_header_regex "([a-zA-Z0-9_-]+): ([^\r]+)\r\n")
  set(http_headers_regex "(${http_header_regex})*")

  set(http_method_regex "GET|HEAD|POST|PUT|DELETE|TRACE|CONNECT")
  set(http_request_uri_regex "[^ ]+")
  set(http_request_line_regex "(${http_method_regex}) (${http_request_uri_regex}) (${http_version_regex})\r\n")
  set(http_request_header_regex "(${http_request_line_regex})(${http_headers_regex})")

  set(http_status_code "[0-9][0-9][0-9]")
  set(http_reason_phrase "[^\r]+")
  set(http_response_line_regex "(${http_version_regex}) (${http_status_code}) (${http_reason_phrase})\r\n")
  set(http_response_header_regex "(${http_response_line_regex})(${http_headers_regex})")
endmacro()