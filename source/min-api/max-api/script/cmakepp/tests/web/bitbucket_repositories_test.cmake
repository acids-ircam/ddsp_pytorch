function(test)

    bitbucket_repositories("kkkkkkkkkkkkkkkkkkk")
    ans(res)
    assert(NOT res)

    bitbucket_repositories("AnotherFoxGuy")
    ans(res)
    assert(${res} CONTAINS ror-dependencies)
endfunction()