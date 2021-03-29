## `log_print`
##
##
function(log_print)
  set(limit ${ARGN})

  address_get(log_record)
  ans(entries)

  list(LENGTH entries len)



  if("${limit}_" STREQUAL "_")
    math(EXPR limit "${len}")
  endif()

  if("${limit}" EQUAL 0)
    return()
  endif()

  foreach(i RANGE 1 ${limit})
    list_pop_back(entries)
    ans(entry)
    if(NOT entry)
      break()
    endif()
    message(FORMAT "{entry.type} @ {entry.function}: {entry.message}")
  endforeach()

endfunction()