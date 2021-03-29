function(test)




  set(list)
  list_pad(list 0)
  list(LENGTH list len)
  _message("${list}")
  assert("${len}" EQUAL 0)



  set(list 1 2)
  list_pad(list 0)
  list(LENGTH list len)
  _message("${list}")
  assert("${len}" EQUAL 2)



  set(list)
  list_pad(list 1 x)
  list(LENGTH list len)
  _message("${list}")
  assert("${list}"  EQUALS "x")

  set(list)
  list_pad(list 2 x)
  list(LENGTH list len)
  _message("${list}")
  assert("${list}"  EQUALS x x)


  set(list)
  list_pad(list 0)
  list(LENGTH list len)
  _message("${list}")
  assert("${len}"  EQUALS "0") 

  # this may seem strange but list contains 1 empty element -> length 0 (ambiguous cmake)
  set(list)
  list_pad(list 1)
  list(LENGTH list len)
  _message("${list}")
  assert("${len}"  EQUALS "0") 

  # this again works how one would suspect
  set(list)
  list_pad(list 2)
  list(LENGTH list len)
  _message("${list}")
  assert("${len}"  EQUAL "2") 


  set(list)
  list_pad(list 3)
  list(LENGTH list len)
  _message("${list}")
  assert("${len}"  EQUAL "3") 


  set(list x)
  list_pad(list 2)
  list(LENGTH list len)
  _message("${list}")
  assert("${len}" EQUAL "2")

  set(list x x)
  list_pad(list 2)
  list(LENGTH list len)
  _message("${list}")
  assert("${len}" EQUAL "2")


  set(list x x x)
  list_pad(list 2)
  list(LENGTH list len)
  _message("${list}")
  assert("${len}" EQUAL "3")




endfunction()