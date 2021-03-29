function(test)

  xml_parse_tags("<thetag d=\"\" bt=\"b\" test=\"a\">content stuff</thetag>" thetag)
  ans(res)
  assert(DEREF {res.value} STREQUAL "content stuff")
  assert(DEREF "{res.attrs.d}_" STREQUAL "_")
  assert(DEREF {res.attrs.bt} STREQUAL "b")
  assert(DEREF {res.attrs.test} STREQUAL "a")

endfunction()