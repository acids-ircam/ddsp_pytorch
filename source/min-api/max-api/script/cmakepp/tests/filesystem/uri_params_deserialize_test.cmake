function(test)



  define_test_function(test_uut uri_params_deserialize)
  test_uut("{a:[1,2,{c:3}]}" "a[]=1&a[]=2&a[][c]=3") 
  test_uut("{a:[2,3]}" "a[]=2&a[]=3")
  test_uut("{a:2}" "a=2")
  test_uut("{a:{b:2}}" "a[b]=2")
  test_uut("{a:[1,2]}" "a[0]=1&a[1]=2")


endfunction()