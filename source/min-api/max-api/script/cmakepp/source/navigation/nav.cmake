# a convenience function for navigating maps
# nav(a.b.c) -> returns memver c of member b of map a
# nav(a.b.c 3) ->sets member c of member b of map a to 3 (creating any missing maps along the way)
# nav(a.b.c = d.e.f) -> assignes the value of d.e.f to a.b.c
# nav(a.b.c += d.e) adds the value of d.e to the value of a.b.c
# nav(a.b.c -= d.e) removes the value of d.e from a.b.c
# nav(a.b.c FORMAT "{d.e}@@{d.f}") formats the string and assigns a.b.c to it
# nav(a.b.c CLONE_DEEP d.e.f) clones the value of d.e.f depely and assigns it to a.b.c
function(nav navigation_expression)
  set(args ${ARGN})
  if("${args}_" STREQUAL "_")
    map_navigate(res "${navigation_expression}")
    return(${res})
  endif()

  if("${ARGN}" STREQUAL "UNSET")
    map_navigate_set("${navigation_expression}")
    return()
  endif()


  set(args ${ARGN})
  list_peek_front(args)
  ans(first)

  if("_${first}" STREQUAL _CALL)
    call(${args})
    ans(args)
  elseif("_${first}" STREQUAL _FORMAT)
    list_pop_front( args)
    format("${args}")  
    ans(args)
  elseif("_${first}" STREQUAL _APPEND OR "_${first}" STREQUAL "_+=")
    list_pop_front(args)
    map_navigate(cur "${navigation_expression}")
    map_navigate(args "${args}")
    set(args ${cur} ${args})
  elseif("_${first}" STREQUAL _REMOVE OR "_${first}" STREQUAL "_-=")
    list_pop_front(args)
    map_navigate(cur "${navigation_expression}")
    map_navigate(args "${args}")
    if(cur)
      list(REMOVE_ITEM cur "${args}")
    endif()
    set(args ${cur})
 elseif("_${first}" STREQUAL _ASSIGN OR "_${first}" STREQUAL _= OR "_${first}" STREQUAL _*)
    list_pop_front( args)
    map_navigate(args "${args}")
    
 elseif("_${first}" STREQUAL _CLONE_DEEP)
    list_pop_front( args)
    map_navigate(args "${args}")
    map_clone_deep("${args}")
    ans(args)
 elseif("_${first}" STREQUAL _CLONE_SHALLOW)
    list_pop_front( args)
    map_navigate(args "${args}")
    map_clone_shallow("${args}")
    ans(args)
  endif()

  # this is a bit hacky . if a new var is created by map_navigate_set
  # it is propagated to the PARENT_SCOPE
  string(REGEX REPLACE "^([^.]*)\\..*" "\\1" res "${navigation_expression}")
  map_navigate_set("${navigation_expression}" ${args})
  set(${res} ${${res}} PARENT_SCOPE)

  return_ref(args)
endfunction()