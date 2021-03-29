  function(uri_remove_schemes uri)
    uri("${uri}")
    ans(uri)
    map_tryget(${uri} schemes)
    ans(schemes)
    list_remove(schemes ${ARGN})
    map_set(${uri} schemes)
    list_combine("+" ${schemes})
    ans(scheme)
    map_tryget(${uri} scheme)
    return_ref(uri)
  endfunction()

  function(uri_set_schemes uri)
    uri("${uri}")
    ans(uri)
    


    map_set(${uri} schemes ${ARGN})

    list_combine("+" ${ARGN})
    ans(scheme)

    map_tryget("${uri}" scheme)
    ans(old_scheme)

    map_set("${uri}" scheme "${scheme}")


    map_tryget(${uri} uri)
    ans(uri_string)

    if(NOT old_scheme)
        set(uri_string "${scheme}:${uri_string}" )
    else()
        string(REPLACE "${old_scheme}:" "${scheme}:" uri_string "${uri_string}")
    endif()

    map_set(${uri} uri "${uri_string}")
    return_ref(uri)
  endfunction()

  function(uri_add_schemes uri)

    uri("${uri}")
    ans(uri)

    map_tryget(${uri} schemes)
    ans(schemes)

    set(schemes ${ARGN} ${schemes})
    list_remove_duplicates(schemes)

    uri_set_schemes(${uri} ${schemes})
    return_ans()

  endfunction()