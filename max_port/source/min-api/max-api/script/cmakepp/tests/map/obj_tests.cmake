function(test)

    obj("")
    ans(res)
    assert(NOT res)

    obj("asd awea ca sdas")
    ans(res)
    assert(NOT res)


    obj("{}")
    ans(res)
    assert(res)

    obj("{
    }")
    ans(res)
    assert(res)

endfunction()