function(Test)
    map_new()
    ans(a)
    map_set("${a}" "x" 0)
    map_get("${a}" "x")
    ans(res)
    assert("${res}" STREQUAL 0)

    map_set("${a}" "x" NOTFOUND)
    map_get("${a}" "x")
    ans(res)
    assert("${res}" STREQUAL NOTFOUND)

    map_set("${a}" "x" NO)
    map_get("${a}" "x")
    ans(res)
    assert("${res}" STREQUAL NO)


endfunction()