function(test)


    map_fill("{a:null,b:1,c:2}" "{a:3,d:4}")
    ans(res)
    assertf("{res.a}" STREQUAL "3")
    assertf("{res.b}" STREQUAL "1")
    assertf("{res.c}" STREQUAL "2")
    assertf("{res.d}" STREQUAL "4")


endfunction()