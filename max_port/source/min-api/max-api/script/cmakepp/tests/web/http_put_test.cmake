function(http_put_test)
    ## is not a stable test because of external parameters.
    ## deactivating...

    return()


    fwrite("myfile.txt" "hello world")
    compress("test.tgz" "myfile.txt")
    http_put("http://httpbin.org/put" --file test.tgz --json)
    ans(res)
    assert(res)
    map_tryget(${res} data)
    ans(data)
    fwrite("result.tgz" "${data}")
    ## - does not work because of encoding


    http_put("http://httpbin.org/put" --file myfile.txt --response)
    ans(res)
    assign(data = res.content)
    #message("${data}")
    json_deserialize("${data}")
    ans(data2)
    #json_print(${data2})
    #json_print(${res})
    assertf("{data2.data}" STREQUAL "hello world")


    http_put("http://httpbin.org/put" "{id:'hello world'}" --json)
    ans(res)
    assertf("{res.json.id}" STREQUAL "hello world")

    http_put("http://httpbin.org/put" "{id:'hello world'}" --exit-code --show-progress)
    ans(error)
    assert(NOT error)

    http_put("http://httpbin.org/put" "{id:'hello world'}" --exit-code)
    ans(error)
    assert("${error}" EQUAL "0")

    http_put("http://httpbin.org/put" "{id:'hello world'}")
    ans(res)
    assert("${res}" MATCHES "hello world")

endfunction()