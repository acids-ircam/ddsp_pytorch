function(test)

  message(inconclusive)
  return()
  cmakepp_config(base_dir)
  ans(base_dir)
  scm_which("${base_dir}")
  ans(res)
  assert("${res}" STREQUAL git)


  pushd("hg" --create)
    hg(init)
    scm_which()
    ans(res)
    assert("${res}" STREQUAL "hg")
  popd()


  scm_which()
  ans(res)
  assert("${res}" ISNULL)

endfunction()