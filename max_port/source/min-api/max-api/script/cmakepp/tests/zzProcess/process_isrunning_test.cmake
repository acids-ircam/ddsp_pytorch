function(test)
  

  ## test the isrunning predicate for a process_handle

  

process_timeout(4)
ans(handle)

## act

## 
process_isrunning(${handle})
ans(isrunning)

sleep(5)

process_isrunning(${handle})
ans(stillrunning)


## assert
assert(isrunning)
assert(NOT stillrunning)






endfunction()