function(test)



 uri_decode("%41%42%43%44%45%46%47%48%49%4A")
 ans(res)
 assert("${res}" STREQUAL "ABCDEFGHIJ")

 uri_decode("%61%62%63%64")
 ans(res)
 assert("${res}" STREQUAL "abcd")

 uri_decode("abc%20def")
 ans(res)
 assert(${res} STREQUAL "abc def")


 uri_encode("hello my friend")
 ans(res)
 assert("${res}" STREQUAL "hello%20my%20friend")

 uri_encode("!@#$%^&*(){}[]|'\"")
 ans(res)
 assert("${res}" STREQUAL "!@%23$%25%5E&*()%7B%7D%5B%5D%7C'%22")





endfunction()







