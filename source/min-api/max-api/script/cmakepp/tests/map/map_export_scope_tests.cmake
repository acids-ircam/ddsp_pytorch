function(test)
    # arrange: set a variable and unset another variable
    set(testvar1 hello)
    set(testvar2 hello2)
    set(testvar2)

    # act
    scope_export_map()
    ans(res)

    # assert that the one variable exists and the other does not
    assert(DEREF "{res.testvar1}" STREQUAL "hello")
    assert(DEREF "{res.testvar2}_" STREQUAL "_")

endfunction()