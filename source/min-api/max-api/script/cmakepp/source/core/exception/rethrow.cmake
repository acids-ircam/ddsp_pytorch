## `(<fail:<bool>> )->` 
## 
## rethrows if the last return value was an exception
## else changes nothing
## if you pass true the exception will be treated as a fatal error
macro(rethrow)
  set(___ans "${__ans}")
  is_exception("${__ans}")
  if(__ans)
    if("${ARGN}_" STREQUAL "true_")
      set(ex ${___ans})
      message(FATAL_ERROR FORMAT "fatal exception: '{ex.message}'")
    endif()
    throw("${___ans}")
  endif()
  set(__ans "${___ans}")
endmacro()


