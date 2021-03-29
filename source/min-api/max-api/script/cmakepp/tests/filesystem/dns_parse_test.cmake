function(test)


  function(dns_test dns expected)
    dns_parse("${dns}")
    ans(dns)
    obj("${expected}")
    ans(expected)

    map_iterator(${expected})
    ans(iter)

    while(true)
      map_iterator_break(iter)

      map_tryget(${dns} ${iter.key})
      ans(value)
      assert(EQUALS ${iter.value} ${value})
    endwhile()

  endfunction()

  # password

  dns_test("user:password@localhost.de" "{password:'password'}")

  # host

  dns_test("userinfo@host:231" "{host:'host'}")
  dns_test("www.google.de" "{host:'www.google.de'}")
  dns_test("192.168.0.1" "{host:'192.168.0.1'}")

  # ip

  dns_test("asd" "{ip:null}")
  dns_test("192.168.0.1" "{ip:'192.168.0.1'}")
  dns_test("tobi@192.168.0.1:1423" "{ip:'192.168.0.1'}")

  # labels

  dns_test("www.google.de" "{labels:['www','google','de']}")
  dns_test("s1231231.host.de" "{labels:['s1231231','host','de']}")
  dns_test("asdasd" "{labels:'asdasd'}")


  # domain
  dns_test("www.google.de" "{domain:'google.de'}")
  dns_test("mydomain" "{domain:'mydomain'}")


  # top_label
  dns_test("www.google.de" "{top_label:'de'}")

endfunction()