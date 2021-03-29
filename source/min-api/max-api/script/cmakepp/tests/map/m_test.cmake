function(test)
    mm("{a:1}")
    ans(res)
    assertf("{res.a}" STREQUAL 1)

    mm(asd = "{b:2}")
    ans(res)
    assert("${res}" STREQUAL "${asd}")
    assertf("{asd.b}" STREQUAL "2")

endfunction()