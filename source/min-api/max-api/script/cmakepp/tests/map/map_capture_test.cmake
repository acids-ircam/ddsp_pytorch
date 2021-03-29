function(test)

    set(a 123)
    set(b 234)
    set(c 345)


    map_capture_new(a b c)
    ans(res)
    assert(${res} MAP_MATCHES "{a:123,b:234,c:345}")


    map_capture_new(asd:a bsd=c b --reassign)
    ans(res)
    assert(${res} MAP_MATCHES "{asd:123,bsd:345,b:234}")


endfunction()