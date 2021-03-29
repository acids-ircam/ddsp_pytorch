## download(uri [target] [--progress])
## downloads the specified uri to specified target path
## if target path is an existing directory the files original filename is kept
## else target is treated as a file path and download stores the file there
## if --progress is specified then the download progress is shown
## returns the path of the successfully downloaded file or null
function(download uri)
  set(args ${ARGN})

  set(uri_string "${uri}")
  uri("${uri}")
  ans(uri)


  list_extract_flag(args --progress)
  ans(show_progress)
  if(show_progress)
    set(show_progress SHOW_PROGRESS)
  else()
    set(show_progress)
  endif()

  list_pop_front(args)
  ans(target_path)
  path_qualify(target_path)

  map_tryget("${uri}" file)
  ans(filename)

  if(IS_DIRECTORY "${target_path}")
    set(target_path "${target_path}/${filename}")    
  endif()
  
  file(DOWNLOAD 
    "${uri_string}" "${target_path}" 
    STATUS status 
   # LOG log
    ${show_progress}
    TLS_VERIFY OFF 
    ${args})


  list_extract(status code message)
  if(NOT "${code}" STREQUAL 0)    
    error("failed to download: {message} (code {code})")
    rm("${target_path}")
    return()
  endif()

  return_ref(target_path)
endfunction()