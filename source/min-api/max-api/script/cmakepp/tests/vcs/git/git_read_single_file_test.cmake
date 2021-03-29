function(test)
    pushd("repo" --create)
    git(init)
    fwrite("README.md" "hello world")
    git(add .)
    git(commit -m "initial commit")
    popd()

    git_read_single_file("repo" "" README.md)
    ans(res)

    assert("${res}" STREQUAL "hello world")

    git_read_single_file("https://github.com/AnotherFoxGuy/cmakepp" "master" LICENSE)
    ans(res)

    assert("${res}" MATCHES "The MIT License")
endfunction()