function(uri_parse_scheme uri)
  map_tryget(${uri} scheme)
  ans(scheme)

  string(REPLACE "+" "\;" schemes "${scheme}")
  map_set(${uri} schemes ${schemes})

endfunction()