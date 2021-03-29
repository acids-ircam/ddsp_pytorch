function(test)

    message("Test inconclusive - too undeterministic")
    return()
    ## fail test
    # http_get("http://notahost.tld")
    # ans(res)
    # assert(NOT res)

    http_get("http://httpbin.org/get")
    ans(content)
    assert(content)

    http_get("http://httpbin.org/get?return_value=lalala" --json)
    ans(content)
    assert(content)
    assertf("{content.args.return_value}" STREQUAL "lalala")


    http_get("http://httpbin.org/get" "{val1:1,val2:2}" --json)
    ans(content)
    assertf("{content.args.val1}" EQUAL 1)
    assertf("{content.args.val2}" EQUAL 2)

    ## fails if you have  a router that displays a custom search page
    ## when it cannot resolve the host.  (*** t-online)
    http_get("http://notahost.tld" --silent-fail)
    ans(res)
    assert(NOT res)

    http_get("http://notahost.tld" --response)
    ans(res)

    assert(res)
    assertf(NOT "{res.client_status}" EQUAL "0")
    assertf("{res.content}_" STREQUAL "_")
    assertf("{res.request_url}" STREQUAL "http://notahost.tld")

    http_get("http://notahost.tld" --exit-code)
    ans(error)
    assert(error)

    http_get("http://httpbin.org/get" --response)
    ans(res)

    assert(res)
    assertf("{res.client_status}" EQUAL 0)
    assertf("{res.request_url}" STREQUAL "http://httpbin.org/get")
    assertf("{res.http_status_code}" STREQUAL "200")
    assertf("{res.http_reason_phrase}" STREQUAL "OK")

    http_get("http://httpbin.org/get" --exit-code)
    ans(error)
    assert(NOT error)

endfunction()
