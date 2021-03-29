function(test)


    obj("{a:1,b:2,c:3}")
    ans(themap)
    ans(copy)
    map_unpack(themap)
    assert(${themap.a} EQUAL 1)
    assert(${themap.b} EQUAL 2)
    assert(${themap.c} EQUAL 3)

    assert("${themap}" STREQUAL "${copy}")


endfunction()