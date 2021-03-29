## 
## checks to see if all specified items are in list 
## using list_check_items
## 
function(uri_check_scheme uri)
  uri_coerce(uri)
  map_tryget(${uri} schemes)
  ans(schemes)
  list_check_items(schemes ${ARGN})
  return_ans()
endfunction()