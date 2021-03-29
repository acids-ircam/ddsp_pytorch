## `(<any> [--algorithm <hash algorithm> = "MD5"])-><checksum>`
##
## this function takes any value and generates its hash
## the difference to string hash is that it serializes the specified object 
## which lets you create the hash for the whoile object graph.  
## 
function(checksum_object obj)
  json("${obj}")
  ans(json)
  checksum_string("${json}" ${ARGN})
  return_ans()
endfunction()