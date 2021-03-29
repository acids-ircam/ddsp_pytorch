## `(<list&...>)-><any...>`
##
## returns all possible combinations of the specified lists
## e.g.
## ```
## set(range 0 1)
## list_combinations(range range range)
## ans(result)
## assert(${result} EQUALS 000 001 010 011 100 101 110 111)
## ```
##
function(list_combinations)
  set(lists ${ARGN})
  list_length(lists)
  ans(len)

  if(${len} LESS 1)
    return()
  elseif(${len} EQUAL 1)
    return_ref(${lists})
  elseif(${len} EQUAL 2)
    list_extract(lists __listA __listB)
    set(__result)
    foreach(elementA ${${__listA}})
      foreach(elementB ${${__listB}})
        list(APPEND __result "${elementA}${elementB}")
      endforeach()
    endforeach()
    return_ref(__result)
  else()
    list_pop_front(lists)
    ans(___listA)

    list_combinations(${lists})
    ans(___listB)

    list_combinations(${___listA} ___listB)
    return_ans()
  endif()
endfunction()