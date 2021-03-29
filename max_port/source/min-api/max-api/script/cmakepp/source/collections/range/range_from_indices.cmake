## `(<index:<uint>...>)-><instanciated range...>`
## 
## returns the best ranges from the specified indices
## e.g range_from_indices(1 2 3) -> [1:3]
##     range_from_indices(1 2) -> 1 2
##     range_from_indices(1 2 3 4 5 6 7 8 4 3 2 1 9 6 7) -> [1:8] [4:1:-1] 9 6 7
function(range_from_indices)
  set(range)
  set(prev)
  set(begin -1)
  set(end -1)
  set(increment)
  list(LENGTH ARGN index_count)
  if(${index_count} EQUAL 0)
    return()
  endif() 


  set(indices_in_partial_range)
  foreach(i ${ARGN})
    if("${begin}"  EQUAL -1)
      set(begin ${i})
      set(end ${i})
    endif()


    if(NOT increment)
      math(EXPR increment "${i} - ${begin}")
      if( ${increment} GREATER 0)
        set(increment "+${increment}")
      elseif(${increment} EQUAL 0)
        set(increment)
      endif()
    endif()

    if(increment)
      math(EXPR expected "${end}${increment}")    
    else()
      set(expected ${i})
    endif()


    if(NOT ${expected} EQUAL ${i})
      __range_from_indices_create_range()
      ## end of current range
      set(begin ${i})
      set(increment)
      set(indices_in_partial_range)

    endif()
    set(end ${i}) 
    list(APPEND indices_in_partial_range ${i})
  endforeach()

  __range_from_indices_create_range()
  


  string(REPLACE ";" " " range "${range}")
  #message("res '${range}'")
  return_ref(range)
endfunction()

## helper macro
macro(__range_from_indices_create_range)
    list(LENGTH indices_in_partial_range number_of_indices)
 #   message("done with range: ${begin} ${end} ${increment} ${number_of_indices}")

    if(${number_of_indices} EQUAL 2)
      list(APPEND range "${begin}")
      list(APPEND range "${end}")
    elseif("${begin}" EQUAL "${end}")
      list(APPEND range "${begin}")
    elseif("${increment}" EQUAL 1)
      list(APPEND range "[${begin}:${end}]")
    else()
      math(EXPR increment "0${increment}")
      list(APPEND range "[${begin}:${end}:${increment}]")
    endif()
endmacro()