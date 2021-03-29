function(test)

  timer_start(t1)
  foreach(i RANGE 0 50)
    
  endforeach()
  timer_print_elapsed(t1)

  define_test_function(test_uut uri_parse uri)
  ## helper because older tests are written with parameters in wrong order
  function(test_uri uri expected)
    test_uut("${expected}" "${uri}")
  endfunction()


  # failed a regex
  test_uri("." "{}")


  ## test schemes

  test_uri("asd+basd:" "{scheme:'asd+basd',schemes:['asd','basd'] }")




  test_uut("{params:{hello:'asd'}}" "?hello=asd")
  test_uut("{params:{hello:['asd','bsd']}}" "?hello[]=asd&hello[]=bsd")





## normalized path

test_uri("../asd/../../bsd/csd/dsd/../../esd/./fsd" "{normalized_segments:['..','..','bsd', 'esd', 'fsd']}" )
# examples
#  test_uri("'scheme1+http://user:password@102.13.44.32:234/C:\\Progr%61m Files(x86)/dir number 1\\file.text.txt?asd=23#asd'" "{}")
#  test_uri("https://www.google.de/u/0/mail/?arg1=123&arg2=arg4#readmails some other data" "{}" --print)
#  test_uri("C:\\windows\\path" "{}" --print)

  ## test userinfo

  test_uri("test" "{user_info:null}")
  test_uri("//becker@localhost" "{user_info:'becker'}")
  test_uri("//becker:password@localhost" "{user_info:'becker:password'}")
  
  ## test dns fields

  test_uri("//becker.tobi:asdasd@192.168.0.1:2313/path/to/nirvana" "{password:'asdasd', user_name:'becker.tobi', ip:'192.168.0.1', port:'2313'}") 
  

  ## test authority
  test_uri("test" "{authority:null}")
  test_uri("//www.google.de" "{authority:'www.google.de'}")
  test_uri("http://www.google.de" "{authority:'www.google.de'}")


  
  # test net_path

  test_uri("test" "{net_path:null}")
  test_uri("C:\\test\\path" "{net_path:'/C:/test/path'}") # because file:// is prepended it is a net_path
  test_uri("/test" "{net_path:'/test'}") # because file:// is prepended it is a net_path
  test_uri("http://localhost" "{net_path:'localhost'}")
  test_uri("http://google.de/file.txt" "{net_path:'google.de/file.txt'}")
  test_uri("mailto:becker@google.de" "{net_path:null}")# no not path because no //
  test_uri("scheme:/de/fa" "{net_path:null}")# no not path because no //


  
  ## test normalization

  test_uri("test a b c" "{uri:'test',rest:' a b c'}")
  test_uri("'test a b c'" "{uri:'test%20a%20b%20c'}")
  test_uri("\"test a b c\"" "{uri:'test%20a%20b%20c'}")
  test_uri("<test a b c>" "{uri:'test%20a%20b%20c'}")
  test_uri("C:/test a b c" "{uri:'file:///C:/test',rest:' a b c'}")
  test_uri("C:\\test\\other a b c" "{uri:'file:///C:/test/other',rest:' a b c'}")
  test_uri("/dev/null a b c" "{uri:'file:///dev/null',rest:' a b c'}")
  test_uri("'C:/test a b c'" "{uri:'file:///C:/test%20a%20b%20c'}")
  test_uri("'C:\\test\\other a b c'" "{uri:'file:///C:/test/other%20a%20b%20c'}")
  test_uri("'/dev/null a b c'" "{uri:'file:///dev/null%20a%20b%20c'}")
  test_uri("//sometext" "{uri:'//sometext'}")

  ## test path

  test_uri("test a b c" "{path:'test'}")
  test_uri("'test a b c'" "{path:'test%20a%20b%20c'}")
  test_uri("C:\\test a b c" "{path:'/C:/test'}")
  test_uri("'C:\\test\\path b\\file.exe'" "{path:'/C:/test/path%20b/file.exe'}")
  test_uri("'a/b c/d'" "{path:'a/b%20c/d'}")
  test_uri("/a/b/c" "{path:'/a/b/c'}")
  test_uri("/a/b/c/" "{path:'/a/b/c/'}")
  test_uri("D:/" "{path:'/D:/'}")
  test_uri("\"C:/Program Files(x86)/Microsoft Visual Studio/Common7\"" "{path:'/C:/Program%20Files(x86)/Microsoft%20Visual%20Studio/Common7'}")
  test_uri("https://www.google.de/u/20/view.xmls?asd=32#showme" "{path:'/u/20/view.xmls'}")
  test_uri("somescheme:somepath/a/b/c" "{path:'somepath/a/b/c'}")

  ## test segments
  
  test_uri("test a b c" "{segments:'test'}")
  test_uri("'test a'" "{segments:'test a'}")
  test_uri("https://github.com/test2" "{segments:'test2'}")
  test_uri("https://github.com/test2/test3" "{segments:['test2','test3']}")
  test_uri("mailto:toeb@github.com" "{segments:'toeb@github.com'}")
  test_uri("'C:\\Program Files\\cmake\\bin\\cmake.exe'" "{segments: ['C:','Program Files', 'cmake', 'bin', 'cmake.exe']}")
  test_uri("C:\\" "{segments:'C:'}")
  test_uri("C:/" "{segments:'C:'}")
  test_uri("/" "{segments:null}")

  ## test lastsegment
  
  test_uri("test/a/b/c.txt" "{last_segment:'c.txt'}")
  test_uri("c.txt" "{last_segment:'c.txt'}")
  test_uri("/" "{last_segment:null}")


  ## test file
  test_uri("test.txt" "{file:'test.txt', file_name:'test', extension:'txt'}")
  test_uri("test" "{file:'test', file_name:'test', extension:null}")
  test_uri("test.txt.xml" "{file:'test.txt.xml' , file_name:'test.txt', extension:'xml'}")
  test_uri("/" "{file:null,file_name:null,extension:null}")


endfunction()