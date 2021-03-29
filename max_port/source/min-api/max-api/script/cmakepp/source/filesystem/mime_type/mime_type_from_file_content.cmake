

function(mime_type_from_file_content file)
  path_qualify(file)
  if(NOT EXISTS "${file}")
    return()
  endif()


  file_isserializedcmakefile("${file}")
  ans(is_serializedcmake)
  if(is_serializedcmake)
    return("application/x-serializedcmake")
  endif()


  file_isqmfile("${file}")
  ans(is_qm)
  if(is_qm)
    return("application/x-quickmap")
  endif()

  file_isjsonfile("${file}")
  ans(is_json)
  if(is_json)
    return("application/json")
  endif()



  file_istarfile("${file}")
  ans(is_tar)
  if(is_tar)
    return("application/x-gzip")
  endif()


  return()
endfunction()