macro(check_host url)
 
  # expect webservice to be reachable
  http_get("${url}" --exit-code)
  ans(error)

  if(error)
    message("Test inconclusive webserver unavailable")
    return()
  endif()

endmacro()