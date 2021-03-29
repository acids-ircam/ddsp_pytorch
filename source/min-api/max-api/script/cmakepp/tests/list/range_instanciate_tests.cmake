function(test)


  range_instanciate(5 "3:1:-2")
  ans(res)
  assert(${res} EQUALS 3:1:-2:true:true:2:true)

  range_instanciate(-1 1:3)
  ans(res)
  assert(${res} EQUALS 1:3:1:true:true:3:false)

  range_instanciate(9 "*")
  ans(res)
  assert(${res} EQUALS 0:9:1:true:true:10:false)

  range_instanciate(9 n/4:n*3/4) # 2 3 4 5 6 
  ans(res)
  assert(${res} EQUALS 2:6:1:true:true:5:false)

  range_instanciate(9 n/2)
  ans(res)
  assert(${res} EQUALS 4:4:1:true:true:1:false)

  range_instanciate(9 $+1)
  ans(res)
  assert(${res} EQUALS 9:9:1:true:true:1:false)

  range_instanciate(9 n)
  ans(res)
  assert(${res} EQUALS 9:9:1:true:true:1:false)


  range_instanciate(9 $)
  ans(res)
  assert(${res} EQUALS 8:8:1:true:true:1:false)

  range_instanciate(9 -1)
  ans(res)
  assert(${res} EQUALS 7:7:1:true:true:1:false)

  range_instanciate(9 -1:-5:-1)
  ans(res)
  assert(${res} EQUALS 7:3:-1:true:true:5:true)

  range_instanciate(1 1)
  ans(res)
  assert(${res} EQUALS 1:1:1:true:true:1:false)

  range_instanciate(3 :) # 0 1 2 
  ans(res)
  assert(${res} EQUALS 0:2:1:true:true:3:false) # 2-1+1

endfunction()