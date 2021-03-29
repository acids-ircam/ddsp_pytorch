# parses a hg ref (e.g. result of hg tags ) returning a map
# { name: <identifier>, number:<int>, id:<hash>}
function(hg_parse_ref)
 string(REGEX REPLACE "^_([a-zA-Z0-9_\\.\\/\\-]+)[ ]+([0-9]+):([0-9a-fA-F]+)(.*)$" "\\1;\\2;\\3;\\4" parts "_${ref}")
  map_new()
  ans(ref_struct)
  list_extract(parts name rev_number rev rest)
  if("${rest}" MATCHES "\\(inactive\\)")
    map_set(${ref_struct} inactive true)
  else()
    map_set(${ref_struct} inactive false)
  endif()


  map_set(${ref_struct} name "${name}")
  map_set(${ref_struct} number "${rev_number}")
  map_set(${ref_struct} hash "${rev}")
  return_ref(ref_struct)
endfunction()
