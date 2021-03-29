
# checks the remote uri if a ref exists ref_type can be * to match any
# else it can be tags heads or HEAD
function(git_remote_has_ref uri ref_name ref_type)
  git_remote_ref("${uri}" "${ref_name}" "${ref_type}")
  ans(res)
  if(res)
    return(true)
  else()
    return(false)
  endif()

endfunction()


