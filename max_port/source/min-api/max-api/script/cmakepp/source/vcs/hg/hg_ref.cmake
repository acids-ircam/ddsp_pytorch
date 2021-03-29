
function(hg_ref  search)
  hg_match_refs("${search}")
  ans(res)
  list(LENGTH res len)
  if("${len}" EQUAL 1)
    return(${res})
  endif()
  return()
endfunction()

