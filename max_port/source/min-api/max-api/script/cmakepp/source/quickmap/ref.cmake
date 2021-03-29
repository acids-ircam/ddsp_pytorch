
## ref() -> <address> 
## 
## begins a new reference value and returns its address
## ref needs to be ended via end() call
function(ref)
  if(NOT ARGN STREQUAL "")
    key("${ARGN}")
  endif()
  address_new()
  ans(ref)
  val(${ref})
  stack_push(quickmap ${ref})   
  return_ref(ref)
endfunction()
