## formats an <uri~> to a localpath 
function(uri_to_localpath uri)
  uri("${uri}")
  ans(uri)

  map_tryget("${uri}" normalized_segments)
  ans(segments)

  map_tryget(${uri} leading_slash)
  ans(rooted)

  map_tryget(${uri} trailing_slash)
  ans(trailing_slash)

  map_tryget(${uri} windows_absolute_path)
  ans(windows_absolute_path)

  string_combine("/" ${segments})
  ans(path)

  if(WIN32 AND "${path}" MATCHES "^[a-zA-Z]:")
    # do nothing
  elseif(rooted AND NOT windows_absolute_path)
    set(path "/${path}")
  endif()
  set(path "${path}${trailing_slash}")
  return_ref(path)
endfunction()
