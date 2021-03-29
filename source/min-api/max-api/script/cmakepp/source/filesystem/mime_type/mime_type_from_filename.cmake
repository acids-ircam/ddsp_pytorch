## mime_type_from_filename() -> 
##
## returns the mimetype for the specified filename
##
##
function(mime_type_from_filename file)
  get_filename_component(extension "${file}" EXT)  
  if("${extension}" MATCHES "([^\\.]+)$")
    set(extension "${CMAKE_MATCH_1}")
  endif()
  mime_type_from_extension("${extension}")
  return_ans()
endfunction()