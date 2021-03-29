## writes the specified varargs to the list
## at the beginning of the specified partial range
## fails if the range is a  multi range
## e.g. 
## set(lstB a b c)
## list_range_partial_write(lstB "[]" 1 2 3)
## -> lst== [a b c 1 2 3]
## list_range_partial_write(lstB "[1]" 1 2 3)
## -> lst == [a 1 2 3 c]
## list_range_partial_write(lstB "[1)" 1 2 3)
## -> lst == [a 1 2 3 b c]
  function(list_range_partial_write __lst __range)
    range_parse("${__range}")
    ans(partial_range)
    list(LENGTH partial_range len)
    if("${len}" GREATER 1)
      message(FATAL_ERROR "only partial partial_range allowed")
      return()
    endif()
   # print_vars(partial_range)

    string(REPLACE ":" ";" partial_range "${partial_range}")
    list(GET partial_range 0 begin)
    list(GET partial_range 1 end)

    if("${begin}" STREQUAL "n" AND "${end}" STREQUAL "n")
      set(${__lst} ${${__lst}} ${ARGN} PARENT_SCOPE)
      return()
    endif()

    list_range_remove("${__lst}" "${__range}")

    list(LENGTH ARGN insertion_count)
    if(NOT insertion_count)
      set(${__lst} ${${__lst}} PARENT_SCOPE)
      return()
    endif() 

    list(GET partial_range 6 reverse)
    if(reverse)
      set(insertion_index "${end}")
    else()
      set(insertion_index "${begin}")
    endif()

    list(LENGTH ${__lst} __len)
    if("${insertion_index}" LESS ${__len})
      list(INSERT ${__lst} "${insertion_index}" ${ARGN})
    elseif("${insertion_index}" EQUAL ${__len})
      list(APPEND ${__lst} ${ARGN})
    else()
      message(FATAL_ERROR "list_range_partial_write could not write to index ${insertion_index}")
    endif()


    set(${__lst} ${${__lst}} PARENT_SCOPE)
    return()
  endfunction()