function(test)








  range_parse("1:8):3") # 1 4 7 => 3
  ans(res)
  assert(${res} EQUALS 1:8:3:true:false:3:false)

  range_parse("1:7):3") # 1 4 => 3
  ans(res)
  assert(${res} EQUALS 1:7:3:true:false:2:false)

  range_parse("1:8:3") # 1 4 7 => 3
  ans(res)
  assert(${res} EQUALS 1:8:3:true:true:3:false)

  range_parse("1:7:3") # 1 4 7 => 3
  ans(res)
  assert(${res} EQUALS 1:7:3:true:true:3:false)

  range_parse("1:6:3") # 1 4 => 2
  ans(res)
  assert(${res} EQUALS 1:6:3:true:true:2:false)

  range_parse("1:4:2")# 1 3
  ans(res)
  assert(${res} EQUALS 1:4:2:true:true:2:false)

  range_parse("1:4):2") # 1 3
  ans(res)
  assert(${res} EQUALS 1:4:2:true:false:2:false)

  range_parse("1:3:2")# 1 3
  ans(res)
  assert(${res} EQUALS 1:3:2:true:true:2:false)

  range_parse("1:3):2")
  ans(res)
  assert(${res} EQUALS 1:3:2:true:false:1:false)

  range_parse("1:2:2")
  ans(res)
  assert(${res} EQUALS 1:2:2:true:true:1:false)

  range_parse("4:1:-2") # 4 2
  ans(res)
  assert(${res} EQUALS 4:1:-2:true:true:2:true)

  range_parse("4:1):-2") # 4 2
  ans(res)
  assert(${res} EQUALS 4:1:-2:true:false:2:true)

  range_parse("3:1:-2") # 3 1
  ans(res)
  assert(${res} EQUALS 3:1:-2:true:true:2:true)

  range_parse("3:1):-2") # 3
  ans(res)
  assert(${res} EQUALS 3:1:-2:true:false:1:true)

  range_parse("2:1:-2") # 2
  ans(res)
  assert(${res} EQUALS 2:1:-2:true:true:1:true)


  range_parse("") # defaults to "[n)"
  ans(res)
  assert(${res} EQUALS "n:n:1:true:false:0:false")
  
  range_parse("()") # defaults to (n)
  ans(res)
  assert("${res}" EQUALS n:n:1:false:false:-1:false)  

  range_parse("[)") 
  ans(res)
  assert("${res}" EQUALS n:n:1:true:false:0:false)


  range_parse("(]")
  ans(res)
  assert("${res}" EQUALS n:n:1:false:true:0:false)


  range_parse("[]")
  ans(res)
  assert("${res}" EQUALS n:n:1:true:true:1:false)

  # range_parse("(") # defaults to "(n)"
  # ans(res)
  # assert("${res}" EQUALS n:n:1:false:false:-1:false)
 
  # range_parse(")") # defaults to [n)
  # ans(res)
  # assert("${res}" EQUALS n:n:1:true:false:0:false)

  # range_parse("[") # defaults to [n)
  # ans(res)
  # assert("${res}" EQUALS n:n:1:true:false:0:false)

  # range_parse("]") # defaults to [n]
  # ans(res)
  # assert("${res}" EQUALS n:n:1:true:true:1:false)

  range_parse("1:2:1:false:false")
  ans(res)
  assert("${res}" EQUALS 1:2:1:false:false:0:false)

  range_parse("1:2:1:false:true")
  ans(res)
  assert("${res}" EQUALS 1:2:1:false:true:1:false) # 2-1-1+1 = 1

  range_parse("1:2:1:true:false")
  ans(res)
  assert("${res}" EQUALS 1:2:1:true:false:1:false) # 2 -1 =1

  range_parse("1:2:1:true:true")
  ans(res)
  assert("${res}" EQUALS 1:2:1:true:true:2:false) # 2-1 +1=2

  range_parse("1:3)")
  ans(res)
  assert("${res}" EQUALS 1:3:1:true:false:2:false) # 3-1 = 2

  range_parse("1:3]")
  ans(res)
  assert("${res}" EQUALS 1:3:1:true:true:3:false) # 3-1+1 =3

  range_parse("(1:3")
  ans(res)
  assert("${res}" EQUALS 1:3:1:false:true:2:false) # 3-1-1 = 2

  range_parse("[1:3")
  ans(res)
  assert("${res}" EQUALS 1:3:1:true:true:3:false) # 3-1+1 3

  range_parse("[1:3)")
  ans(res)
  assert("${res}" EQUALS 1:3:1:true:false:2:false) # 3-1=2

  range_parse("[1:3]")
  ans(res)
  assert("${res}" EQUALS 1:3:1:true:true:3:false) # 3-1+1

  range_parse("(1:3]")
  ans(res)
  assert("${res}" EQUALS 1:3:1:false:true:2:false) # 3-1 +1-1 =2

  range_parse("(1:3)")
  ans(res)
  assert("${res}" EQUALS 1:3:1:false:false:1:false)# 3-1 -1 = 1


  range_parse("1:3:1:false:true:4") #length is ignored
  ans(res)
  assert(${res} EQUALS 1:3:1:false:true:2:false)#3-1+1-1=2


  range_parse(":")
  ans(res)
  assert(${res} EQUALS  0:$:1:true:true:$-0+1:false)

  range_parse("(:)")
  ans(res)
  assert(${res} EQUALS 0:$:1:false:false:$-0-1:false)

  range_parse("$")
  ans(res)
  assert(${res} EQUALS  $:$:1:true:true:1:false)

  range_parse("n")
  ans(res)
  assert(${res} EQUALS n:n:1:true:true:1:false)

  range_parse("1")# defaults to [1]
  ans(res)
  assert(${res} EQUALS  1:1:1:true:true:1:false)

  range_parse("1:3") # defaults to [1:3]
  ans(res)
  assert(${res} EQUALS  1:3:1:true:true:3:false) # 3-1+1=3

  range_parse("3:1")
  ans(res)
  assert(${res} ISNULL)

  range_parse("1:3:-1")
  ans(res)
  assert(${res} ISNULL)

  range_parse("1;2;3")
  ans(res)
  assert(${res} EQUALS  1:1:1:true:true:1:false  2:2:1:true:true:1:false  3:3:1:true:true:1:false)


  range_parse("[3:1]:-1") # 3,2,1
  ans(res)
  assert(${res} EQUALS 3:1:-1:true:true:3:true) # 

  range_parse("(3:1):-1") # 2
  ans(res)
  assert(${res} EQUALS 3:1:-1:false:false:1:true) # 


  range_parse("4:15:3") #4,7,10,13
  ans(res)
  assert(${res} EQUALS 4:15:3:true:true:4:false)


  range_parse("1:3 6:4:-1 2 9:4:3 : $")
  ans(res)
  assert(${res} EQUALS 
   1:3:1:true:true:3:false  
   6:4:-1:true:true:3:true
   2:2:1:true:true:1:false  
   0:$:1:true:true:$-0+1:false  
   $:$:1:true:true:1:false
   )


  range_parse("-0")
  ans(res)
  assert(${res} EQUALS  $:$:1:true:true:1:false)


  range_parse("-1:-3:-1")
  ans(res)
  assert(${res} EQUALS  "($-1):($-3):-1:true:true:($-1)-($-3)+1:true")
endfunction()