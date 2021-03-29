function(test)

uri_check_scheme(":" bitbucket?)
ans(res)
assert(res)

uri_check_scheme("bitbucket:" bitbucket?)
ans(res)
assert(res)


uri_check_scheme("bitbucket+github:" bitbucket?)
ans(res)
assert(NOT res)


uri_check_scheme("a+b+c:" a b? c)
ans(res)
assert(res)

uri_check_scheme("a+b+c:" a !c)
ans(res)
assert(NOT res)


uri_check_scheme("a+b+c:" a? b? c?)
ans(res)
assert(res)

uri_check_scheme("a+b+c:")
ans(res)
assert(NOT res)

uri_check_scheme("a+b+c:" a? b c? !d)
ans(res)
assert(res)


endfunction()