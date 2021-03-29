function(uri_parse_authority uri)
  map_get(${uri} authority)
  ans(authority)

  map_get(${uri} net_path)
  ans(net_path)

  ## set authoirty to localhost if no other authority is specified but it is a net_path (starts wth //)
  if("_authority" STREQUAL "_" AND NOT "${net_path}_" STREQUAL "_")
    set(authority localhost)
  endif()

  dns_parse("${authority}")
  ans(dns)


  map_iterator(${dns})
  ans(it)
  while(true)
    map_iterator_break(it)
    if(NOT "${it.key}" STREQUAL "rest")
      map_set(${uri} ${it.key} ${it.value})
    endif()
  endwhile()

  return()

endfunction()
