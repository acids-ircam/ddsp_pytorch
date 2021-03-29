function(test)

    map_from_keyvaluelist("" "k1" "v1 v2 v3" "k2" "v4 v5 v6")
    ans(res)

    assert(DEREF "{res.k1}" STREQUAL "v1 v2 v3")
    assert(DEREF "{res.k2}" STREQUAL "v4 v5 v6")


    map_to_keyvaluelist(${res})
    ans(asd)

    assert(EQUALS "${asd}" "k1;v1 v2 v3;k2;v4 v5 v6")


    set(uut "hello, world { I am } b [ ]")
    string_take(uut "hello, ")
    ans(res)
    assert("${uut}" STREQUAL "world { I am } b [ ]")
    assert("${res}" STREQUAL "hello, ")

    string_take_regex(uut "[^ ]+ ")
    ans(res)
    assert("${res}" STREQUAL "world ")
    assert("${uut}" STREQUAL "{ I am } b [ ]")

    string_take_regex(uut asd)
    ans(res)
    assert("${res}_" STREQUAL "_")
    assert("${uut}" STREQUAL "{ I am } b [ ]")

    string_take(uut asdf)
    ans(res)
    assert("${res}_" STREQUAL "_")
    assert("${uut}" STREQUAL "{ I am } b [ ]")


    set(uut "[123")
    string_take_regex(uut "0|[1-9][0-9]*")
    ans(res)
    assert("${res}_" STREQUAL "_")
    return()
    string(ASCII 28 file_separator)
    string(ASCII 29 group_separator)
    string(ASCII 30 record_separator)
    string(ASCII 31 unit_separator)
    string(ASCII 31 unit_separator)
    foreach (i RANGE 1 32)
        string(ASCII ${i} char)
        message("${i}: ${char}")
    endforeach ()


    #  GS key1 US
    #    val1

    #  GS key2 US US
    #    val2
    #  GS key3 US
    #    val31 US val32 US val33


    function(map_ascii_from str)


    endfunction()
    function(map_ascii_to)

    endfunction()
endfunction()