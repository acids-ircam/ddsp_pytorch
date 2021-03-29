

# converts the <structured data?!...> into  <structured data...>
function(objs)
  set(res)
  foreach(arg ${ARGN})
    obj(${arg})
    ans(arg)
    list(APPEND res "${arg}")
  endforeach()
  return_ref(res)
endfunction()