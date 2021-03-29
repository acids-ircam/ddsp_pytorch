## `(<length:<int>> <~range...>)-><index:<uint>...>` 
##
## returns the list of indices for the specified range
## length may be negative which causes a failure if any anchors are used (`$` or `n`) 
## 
## if the length is valid  (`>-1`) only valid indices are returned or failure occurs
##
## a length of 0 always returns no indices
##
## **Examples**
## ```
## ```
function(range_indices length)

  if("${length}" EQUAL 0)
    return()
  endif()
  if("${length}" LESS 0)
    set(length 0)
  endif()
  
  range_instanciate("${length}" ${ARGN})
  ans(range)

  ## foreach partial range in range 
  ## get the begin and end and increment 
  ## use cmake's foreach loop to enumerate the range 
  ## and save the indices 
  ## remove a index at front and or back if the inclusivity warrants it
  ## return the indices
  set(indices)
  foreach(partial ${range})
    string(REPLACE ":" ";" partial "${partial}")
    list(GET partial 0 1 2 partial_range)
    foreach(i RANGE ${partial_range})
      list(APPEND indices ${i})
    endforeach() 
    list(GET partial 3 begin_inclusivity)
    list(GET partial 4 end_inclusivity)
    if(NOT end_inclusivity)
      list_pop_back(indices)
    endif()
    if(NOT begin_inclusivity)
      list_pop_front(indices)
    endif()
  endforeach()
  return_ref(indices)
endfunction()
