function(test)

    map_navigate_set("a.b.c" 3)
    assert(DEREF "{a.b.c}" STREQUAL "3")

    map_navigate_set("a.b.d" 4)
    map_navigate_set("a.a" 5)
    assert(DEREF "{a.b.c}" STREQUAL "3")
    assert(DEREF "{a.b.d}" STREQUAL "4")
    assert(DEREF "{a.a}" STREQUAL "5")

endfunction()