function(test)

  map_new()
  ans(map)
  path_to_map("${map}" "/home/mbslib/Documents/test")
  assert(DEREF "{map.home.mbslib.Documents.test}" STREQUAL "/home/mbslib/Documents/test")


endfunction()