
  function(uri_format uri)
    set(args ${ARGN})

    list_extract_flag(args --no-query)
    ans(no_query)

    list_extract_flag(args --no-scheme)
    ans(no_scheme)

    list_extract_labelled_value(args --remove-scheme)
    ans(remove_scheme)



    obj("${args}")
    ans(payload)


    uri("${uri}")
    ans(uri)
    map_tryget("${uri}" params)
    ans(params)

    if(payload)

      map_merge( "${params}" "${payload}")
      ans(params)
    endif()

    set(query)
    if(NOT no_query)
      uri_params_serialize("${params}")
      ans(query)
      if(query)
        set(query "?${query}")
      endif()
    endif()

    if(NOT no_scheme)

      if(NOT remove_scheme STREQUAL "")
        map_tryget("${uri}" schemes)
        ans(schemes)

        string(REPLACE "+" ";" remove_scheme "${remove_scheme}")

        list_remove(schemes ${remove_scheme})
        string_combine("+" ${schemes})
        ans(scheme)
      else()
        map_tryget("${uri}" scheme)
        ans(scheme)
      endif()

      if(NOT "${scheme}_" STREQUAL "_")
        set(scheme "${scheme}:")
      endif()
    endif()

    map_tryget("${uri}" net_path)
    ans(net_path)

    if("${net_path}_" STREQUAL "_")
      map_tryget(${uri} path)
      ans(path)
      set(uri_string "${scheme}${path}${query}")
    else()
      set(uri_string "${scheme}//${net_path}${query}")
    endif()
    return_ref(uri_string)

  endfunction()