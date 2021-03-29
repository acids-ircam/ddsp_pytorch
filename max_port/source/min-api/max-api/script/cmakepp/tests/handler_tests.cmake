function(test)
  function(testhandler2 in)
    return("__${in}__")
  endfunction()

  handler_default(testhandler2)
  ans(handler)
  handler_execute("${handler}" "{input:'asd'}")
  ans(res)
  assert_matches("{output:'__asd__'}" res)



  ## handler find test
  assign(handlers = "[
{callable: 'test_func', display_name: 'command1', id:'1', labels:'cmd1'},
{callable: 'test_func', display_name: 'command2', id:'2', labels:'cmd2'},
{callable: 'test_func', display_name: 'command3', id:'3', labels:['cmd3','cmd_3']}
    ]")

  #assign(result = handler_find(handlers "{input:['cmd2','b','c']}"))

  assert_matches("handlers[1]" handler_find(handlers "{input:['cmd2','b','c']}"))



  ## handler execute test
  function(test_func request response)
    assign(response.output += request.input)
  endfunction()

  ## valid handler func
  assert_matches("{output:'asdasdf',handler:{callable:'test_func'}}" handler_execute("{callable:'test_func'}" "{input:'asdf'}" "{output:'asd'}"))
  ## invalid handler func
 ## assert_matches("{output:'asd', error:'handler_invalid'}" handler_execute("{callable:'non_existent_func'}" "{input:'asdf'}" "{output:'asd'}"))


  ## on the fly handler
  handler_execute("[](req res)map_set({{res}} output yaaay)" "{}")
  ans(res)
  assert_matches("{output:'yaaay'}" res)

endfunction()