function(test)

    data(uut.find_handler (command1 "{a:123}"))
    ans(res)
    assert(${res} MAP_MATCHES "['uut.find_handler','(','command1',{a:123},')']")


endfunction()