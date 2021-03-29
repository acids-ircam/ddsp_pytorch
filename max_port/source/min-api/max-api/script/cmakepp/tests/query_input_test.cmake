function(test)




  address_set(myref a b c d e f g h i j k )
  function(test_query_input)
    address_pop_front(myref)
    ans(res)
    message_indent_push(0)
    message("> ${res}")
    message_indent_pop()
    return_ref(res)
  endfunction()


  type_def("{
    type_name:'address',
    properties:[
      'street:string',
      'area_code:string'
    ]
  }")
  



  query_type(test_query_input "first_name:string;last_name;address:address;bla:['a','b','c']")
  ans(res)
  json_print(${res})

endfunction()