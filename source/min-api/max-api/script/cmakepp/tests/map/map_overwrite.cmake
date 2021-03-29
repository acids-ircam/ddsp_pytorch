function(test)


    map_overwrite("{a:1,b:2,c:3}" "{a:4,c:5,d:6}")
    ans(res)


    assertf("{res.a}" STREQUAL "4")
    assertf("{res.b}" STREQUAL "2")
    assertf("{res.c}" STREQUAL "5")
    assertf("{res.d}" STREQUAL "6")

endfunction()