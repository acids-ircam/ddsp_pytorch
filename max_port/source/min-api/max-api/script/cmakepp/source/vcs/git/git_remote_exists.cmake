
# checks wether the uri is a remote git repository
function(git_remote_exists uri)
  git_uri("${uri}")
  ans(uri)


  git_lean(ls-remote "${uri}")
  ans_extract(error)
  
  if(error)
    return(false)
  endif()
  return(true)
endfunction()