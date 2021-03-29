# returns true iff the uri is a hg repository
function(hg_remote_exists uri)
  hg(identify "${uri}" --exit-code)
  ans(error)

  if(NOT error)
    return(true)
  endif()
  return(false)
endfunction()
