function(test)

    script("{
    k1:'v1',
    k2:'v2',
    k3:{
      k4:'v4',
      k5:'v5',
      k6:{
        k7:'v7',
        k8:'v8'
      }
    }
  }")
    ans(uut)

    # element(MAP)
    #   value(KEY "k1" "v1")
    #   value(KEY "k2" "v2")
    #   element(MAP k3)
    #     value(KEY "k4" "v4")
    #     value(KEY "k5" "v5")
    #     element(MAP k6)
    #       value(KEY "k7" "v7")
    #       value(KEY "k8" "v8")
    #     element(END)
    #   element(END)
    # element(END uut)

    map_clone(${uut} SHALLOW)
    ans(cloned)

    assert(cloned)
    assert(NOT "${cloned}" STREQUAL "${uut}")
    assert(DEREF "{cloned.k1}" STREQUAL "{uut.k1}")
    assert(DEREF "{cloned.k2}" STREQUAL "{uut.k2}")
    assert(DEREF "{cloned.k3}" STREQUAL "{uut.k3}")


    map_clone(${uut} DEEP)
    ans(cloned)
    assert(cloned)
    assert(NOT "${cloned}" STREQUAL "${uut}")
    assert(DEREF "{cloned.k1}" STREQUAL "{uut.k1}")
    assert(DEREF "{cloned.k2}" STREQUAL "{uut.k2}")
    assert(DEREF NOT "{cloned.k3}" STREQUAL "{uut.k3}")
    assert(DEREF "{cloned.k3.k4}" STREQUAL "{uut.k3.k4}")
    assert(DEREF "{cloned.k3.k5}" STREQUAL "{uut.k3.k5}")
    assert(DEREF NOT "{cloned.k3.k6}" STREQUAL "{uut.k3.k6}")
    assert(DEREF "{cloned.k3.k6.k7}" STREQUAL "{uut.k3.k6.k7}")
    assert(DEREF "{cloned.k3.k6.k8}" STREQUAL "{uut.k3.k6.k8}")


    map_clone("" DEEP)
    ans(cloned)
    assert(NOT cloned)

    map_clone("" SHALLOW)
    ans(cloned)
    assert(NOT cloned)

    map_clone("hello there" DEEP)
    ans(cloned)
    assert(${cloned} STREQUAL "hello there")

    map_clone("hello there" SHALLOW)
    ans(cloned)
    assert(${cloned} STREQUAL "hello there")

    map_clone("list;value" DEEP)
    ans(cloned)
    assert(EQUALS ${cloned} list value)

    map_clone("list;value" SHALLOW)
    ans(cloned)
    assert(EQUALS ${cloned} list value)

    address_new()
    ans(uut_ref)
    address_set(${uut_ref} hello)

    map_clone("${uut_ref}" SHALLOW)
    ans(cloned)
    assert(cloned)
    assert(NOT ${cloned} STREQUAL ${uut_ref})
    address_get(${cloned})
    ans(res)
    assert(DEREF "${res}" STREQUAL "hello")

    map_clone("${uut_ref}" DEEP)
    ans(cloned)
    assert(cloned)
    assert(NOT ${cloned} STREQUAL ${uut_ref})
    address_get(${cloned})
    ans(res)
    assert(DEREF "${res}" STREQUAL "hello")


endfunction()