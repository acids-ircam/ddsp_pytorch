function(test)
  pushd(repo1 --create)
  git(init)
  popd()


  # remote web repo
  git_remote_exists("https://github.com/AnotherFoxGuy/cmakepp.git")
  ans(res)
  assert(res)


  # non existing repo in existing path
  git_remote_exists("${test_dir}")
  ans(res)
  assert(NOT res)

  # existing repo in existing path
  git_remote_exists("${test_dir}/repo1")
  ans(res)
  assert(res)

  # existing repo in exisiting relative path
  cd("${test_dir}")
  git_remote_exists("repo1")
  ans(res)
  assert(res)

  # bogus string
  git_remote_exists("nananana")
  ans(res)
  assert(NOT res)



  
endfunction()