function(test)


    # element(MAP)
    #   value(KEY k1 v1)
    #   value(KEY k2 v2)
    #   value(KEY k3 v3)
    #   value(KEY k4 v4)
    # element(END uut)

    obj("{
  k1:'v1',
  k2:'v2',
  k3:'v3',
  k4:'v4'
}")
    ans(uut)

    set(c c)
    set(d d)
    set(e e)
    map_extract("uut.k1;uut.k3" a b c d e)
    assert(${a} STREQUAL "v1")
    assert(${b} STREQUAL "v3")
    assert(NOT c)
    assert(NOT d)
    assert(NOT e)

endfunction()