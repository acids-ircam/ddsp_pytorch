function(test)
    # creates two maps and merges them
    # create two maps
    map()
    kv(k1 1)
    kv(k2 1)
    end()
    ans(uut1)

    map()
    kv(k2 2)
    kv(k3 2)
    end()
    ans(uut2)

    #merge by overwriting map2 with map1
    map_new()
    ans(res)
    map_union(${res} ${uut2} ${uut1})

    assert(res)
    map_keys(${res})
    ans(keys)
    map_tryget(${res} k1)
    ans(val)
    assert(${val} STREQUAL 1)

    map_tryget(${res} k2)
    ans(val)
    assert(${val} STREQUAL 1)

    map_tryget(${res} k3)
    ans(val)
    assert(${val} STREQUAL 2)

    # merge in oposite direction
    map_new()
    ans(res)
    map_union(${res} ${uut1} ${uut2})
    assert(res)
    map_tryget(${res} k1)
    ans(val)
    assert(${val} STREQUAL 1)

    map_tryget(${res} k2)
    ans(val)
    assert(${val} STREQUAL 2)

    map_tryget(${res} k3)
    ans(val)
    assert(${val} STREQUAL 2)

    # check if deep elements are merged
    obj("{elem1:{k1:'v1'}}")
    ans(m1)

    obj("{elem1:{k2:'v2'}}")
    ans(m2)


    set(res)
    map_merge(${m1} ${m2})
    ans(res)
    assert(DEREF "{res.elem1.k1}" STREQUAL "v1")
    assert(DEREF "{res.elem1.k2}" STREQUAL "v2")
endfunction()