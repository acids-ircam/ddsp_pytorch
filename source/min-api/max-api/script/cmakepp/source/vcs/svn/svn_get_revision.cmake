## returns the revision for the specified svn uri
function(svn_get_revision)
  svn_info("${ARGN}")
  ans(res)
  nav(res.revision)
  return_ans()
endfunction()