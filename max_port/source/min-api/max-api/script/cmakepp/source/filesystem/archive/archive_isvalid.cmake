## `archive_isvalid(<path>)-> <bool>`
##
## returns true if the specified path identifies an archive 
## file
function(archive_isvalid file)
  mime_type("${file}")
  ans(types)

  list_contains(types "application/x-gzip")
  ans(is_archive)


  return_ref(is_archive)
endfunction()