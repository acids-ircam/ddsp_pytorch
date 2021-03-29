function(test)

    message("Test inconclusive")
    return()

    map()
    kv(a 1)
    kv(b 2)
    kv(c 3)
    end()
    ans(map)


    address_new()
    ans(myref)
    map_foreach("${map}" "[](key val)address_append({{myref}} {{key}} {{val}})")
    address_get(${myref})
    ans(vals)

    assert(EQUALS ${vals} a 1 b 2 c 3)

    map_foreach("" "[]() ")


endfunction()