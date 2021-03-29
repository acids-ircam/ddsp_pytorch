
# orders a list by a comparator function
function(list_sort __list_order_lst comparator)
  list(LENGTH ${__list_order_lst} len)

  function_import("${comparator}" as __compare REDEFINE)

  # copyright 2014 Tobias Becker -> triple s "slow slow sort"
  set(i 0)
  set(j 0)
  while(true)
    if(NOT ${i} LESS ${len})
      set(i 0)
      math(EXPR j "${j} + 1")
    endif()

    if(NOT ${j} LESS ${len}  )
      break()
    endif()
    list(GET ${__list_order_lst} ${i} a)
    list(GET ${__list_order_lst} ${j} b)
    #rcall(res = "${comparator}"("${a}" "${b}"))
    __compare("${a}" "${b}")
    ans(res)
    if(res LESS 0)
      list_swap(${__list_order_lst} ${i} ${j})
    endif()


    math(EXPR i "${i} + 1")
  endwhile()
  return_ref(${__list_order_lst})
endfunction()

## faster implementation: quicksort


# orders a list by a comparator function and returns it
function(list_sort __list_sort_lst comparator)
  list(LENGTH ${__list_sort_lst} len)
  math(EXPR len "${len} - 1")
  function_import("${comparator}" as __quicksort_compare REDEFINE)
  __quicksort(${__list_sort_lst} 0 ${len})
  return_ref(${__list_sort_lst})
endfunction()

   ## the quicksort routine expects a function called 
   ## __quicksort_compare to be defined
 macro(__quicksort __list_sort_lst lo hi)
  if("${lo}" LESS "${hi}")
    ## choose pivot
    set(p_idx ${lo})
    ## get value of pivot 
    list(GET ${__list_sort_lst} ${p_idx} p_val)
    
    list_swap(${__list_sort_lst} ${p_idx} ${hi})
    math(EXPR upper "${hi} - 1")
    
    ## store index p
    set(p ${lo})
    foreach(i RANGE ${lo} ${upper})
      list(GET ${__list_sort_lst} ${i} c_val)
      __quicksort_compare("${c_val}" "${p_val}")
      ans(cmp)
      if("${cmp}" GREATER 0)
        list_swap(${__list_sort_lst} ${p} ${i})
        math(EXPR p "${p} + 1")
      endif()
    endforeach()
    list_swap(${__list_sort_lst} ${p} ${hi})

    math(EXPR p_lo "${p} - 1")
    math(EXPR p_hi "${p} + 1")
    ## recursive call
    __quicksort("${__list_sort_lst}" "${lo}" "${p_lo}")
    __quicksort("${__list_sort_lst}" "${p_hi}" "${hi}")
  endif()
 endmacro()
