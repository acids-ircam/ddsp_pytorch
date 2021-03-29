

## appends the last return value to the specified list
macro(ans_append __lst)
  list(APPEND ${__lst} ${__ans})
endmacro()