

## returns the mimetyoe object for the specified name or extension
function(mime_type_get name_or_ext)
  mime_type_map()
  ans(mm)
  map_tryget("${mm}" "${name_or_ext}")
  return_ans()
endfunction()