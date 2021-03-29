## returns the file type for the specified file
## only existing files can have a file type
## if an existing file does not have a specialized file type
## the extension is returned
function(mime_type file)
  path_qualify(file)

  if(NOT EXISTS "${file}")
    #message("no file")
    return(false)
  endif()

  if(IS_DIRECTORY "${file}")
   # message("is dir")
    return(false)
  endif()


  mime_type_from_file_content("${file}")
  ans(mime_type)

  if(mime_type)
    return_ref(mime_type)
  endif()



  mime_type_from_filename("${file}")
  ans(mime_type)


  return_ref(mime_type)
  
endfunction()
