function(test)
    git(--version)
    ans(res)
    assert("${res}" MATCHES "git")

    pushd("dir1" --create)
    git(init --exit-code)
    ans(ret)
    assert("${ret}" EQUAL 0)
    assert(EXISTS "${test_dir}/dir1/.git")

    ## create a repo and add a file to it
    fwrite(README.md "hello world")
    git(add . --exit-code)
    ans(ret)
    assert("${ret}" EQUAL 0)
    git(commit -m "initial commit" --exit-code)
    ans(ret)
    assert("${ret}" EQUAL 0)

    popd()

    ## clone local repo
    pushd(dir2 --create)
    git(clone ../dir1 . --exit-code)
    ans(ret)
    assert("${ret}" EQUAL 0)
    assert(EXISTS "${test_dir}/dir2/README.md")
    popd()


    ## clone remote repo (of cmakepp)
    pushd(dir3 --create)
    git(clone "https://github.com/AnotherFoxGuy/cmakepp" . --exit-code)
    ans(ret)
    assert("${ret}" EQUAL 0)
    assert(EXISTS "${test_dir}/dir3/source/vcs/git/git.cmake")

endfunction()