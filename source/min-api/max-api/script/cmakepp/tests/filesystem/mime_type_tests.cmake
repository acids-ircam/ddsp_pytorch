function(test)
  
  mime_type_register("{
    name:'application/x-custom1',
    extensions:'c1'
  }")
  ans(res)
  assert(res)


  mime_type_from_extension(".c1")
  ans(res)
  assert("${res}" STREQUAL "application/x-custom1")


  mime_type_from_extension(".xml")
  ans(res)
  assert("${res}" STREQUAL "application/xml")

  mime_type_get("json")
  ans(res)
  assert(res)
  assertf("{res.description}" STREQUAL "JavaScript Object Notation")



    mime_type_get_extension("application/x-cmake")
    ans(ext)
    assert("${ext}" STREQUAL "cmake")


    mime_type_get_extension("application/json")
    ans(ext)
    assert("${ext}" STREQUAL "json")


    mime_type_get_extension("application/x-quickmap")
    ans(ext)
    assert("${ext}" STREQUAL "qm")


    mime_type_get_extension("application/x-gzip")
    ans(ext)
    assert("${ext}" STREQUAL "tgz")


    mime_type_get_extension("text/plain")
    ans(ext)
    assert("${ext}" STREQUAL "txt")


endfunction()