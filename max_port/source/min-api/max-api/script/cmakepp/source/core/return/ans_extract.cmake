## extracts the the specified variables in order from last result
## returns the rest of the result which was unused
## ```
## do_something()
## ans_extract(value1 value2)
## ans(rest)
## ``` 
macro(ans_extract)
  ans(__ans_extract_list)
  list_extract(__ans_extract_list ${ARGN})
endmacro()