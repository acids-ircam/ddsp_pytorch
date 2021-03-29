

## returns a map of known mime types
function(mime_type_map)
  map_new()
  ans(mime_type_map)
  map_set(global mime_types "${mime_type_map}")

  function(mime_type_map)
    map_tryget(global mime_types)
    return_ans()
  endfunction()

  mime_types_register_default()



  mime_type_map()
  return_ans()
endfunction()