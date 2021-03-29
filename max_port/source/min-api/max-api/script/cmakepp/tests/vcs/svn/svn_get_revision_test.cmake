function(test)

  find_package(Subversion)
  if(NOT SUBVERSION_FOUND)
    message("test inconclusive svn not installed")
    return()
  endif()

  svn_info("http://llvm.org/svn/llvm-project/llvm/trunk")
  ans(res)
  assert(res)

  svn_get_revision("http://llvm.org/svn/llvm-project/llvm/trunk")
  ans(res)
  assert(res)

endfunction()