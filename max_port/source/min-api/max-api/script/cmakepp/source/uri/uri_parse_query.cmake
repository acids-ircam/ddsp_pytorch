## parses the query field of uri and sets  the uri.params field to the parsed data
function(uri_parse_query uri)
  map_tryget(${uri} query)
  ans(query)
  uri_params_deserialize("${query}")
  ans(params)
  map_set(${uri} params ${params})
  return()

endfunction()