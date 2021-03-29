function(test)


    # multivalue
    map_equal("123;456" "123;456")
    ans(res)
    assert(res)

    map_equal("123;456" "123;678")
    ans(res)
    assert(NOT res)


    # with cycle
    map_new()
    ans(lhs)
    map_set(${lhs} a ${lhs})
    map_new()
    ans(rhs)
    map_set(${rhs} a ${rhs})

    map_equal(${lhs} ${rhs})
    ans(res)
    assert(res)


    # with cross cycle
    map_new()
    ans(lhs)

    map_new()
    ans(rhs)

    map_set(${lhs} a ${rhs})
    map_set(${rhs} a ${lhs})

    map_equal(${lhs} ${rhs})
    ans(res)
    assert(res)


    map_new()
    ans(uut)
    map_append(${uut} k1 a)
    map_append(${uut} k1 b)
    map_append(${uut} k1 c)

    map_append(${uut} k2 c)
    map_append(${uut} k2 c)

    map_append(${uut} k3 "asd asda asd")

    map_new()
    ans(uut2)
    map_append(${uut2} k2 c)
    map_append(${uut2} k2 c)
    map_append(${uut2} k1 a)
    map_append(${uut2} k1 b)
    map_append(${uut2} k1 c)


    map_append(${uut2} k3 "asd asda asd")


    map_equal(${uut} ${uut2})
    ans(res)
    assert(res)

    map_equal(${uut2} ${uut})
    ans(res)
    assert(res)

    map_set(${uut} k1 "ello")

    map_equal(${uut} ${uut2})
    ans(res)
    assert(NOT res)


    map_equal(${uut2} ${uut})
    ans(res)
    assert(NOT res)


    map_new()
    ans(lhs)

    map_new()
    ans(rhs)

    # null map
    map_equal(${lhs} ${rhs})
    ans(res)
    assert(res)

    map()

    key(c)
    map()
    kv(k k)
    end()
    val(d)
    map()
    kv(f f)
    end()

    kv(a a)
    map(b)
    kv(c c)
    end()

    end()
    ans(lhs)

    map()
    map(b)
    kv(c c)
    end()
    kv(a a)
    key(c)
    map()
    kv(k k)
    end()
    val(d)
    map()
    kv(f f)
    end()


    end()
    ans(rhs)

    map_equal(${lhs} ${rhs})
    ans(res)
    assert(res)


endfunction()