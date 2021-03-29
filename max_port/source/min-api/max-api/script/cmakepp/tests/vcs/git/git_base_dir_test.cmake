function(test)
cd("${test_dir}")
  pushd("repo" --create)
  git(init)
  pushd("a/b/c" --create)  
  git_base_dir()
  ans(res)
  popd()
  popd()

  assert("${res}" STREQUAL "${test_dir}/repo")

## fails because sometimes tmp dir is in repo
  #git_base_dir()
  #ans(res)
  #assert(NOT res)
endfunction()