function(test)
  pwd()
  ans(res)
  message("${res}")

  pushd("repo" --create)
    git(init)
    fwrite(README.md "hello world")
    git(add .)
    git(commit -m "initial commit")    
  popd()


  git_remote_has_ref("repo" HEAD *)
  ans(res)
  assert(res)

  git_remote_has_ref("repo" master *)
  ans(res)
  assert(res)

  git_remote_has_ref("repo" nonono *)
  ans(res)
  assert(NOT res)




endfunction()