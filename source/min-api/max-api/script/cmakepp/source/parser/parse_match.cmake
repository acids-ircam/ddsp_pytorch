
  function(parse_match rstring)
    address_get(${rstring})
    ans(str)

    map_get(${definition} search)
    ans(search)

   # message("parsing match with '${parser_id}' (search: '${search}') for '${str}'")
    map_tryget(${definition} ignore_regex)
    ans(ignore_regex)
   #message("ignore: ${ignore_regex}")
    list(LENGTH ignore_regex len)
    if(len)
     # message("ignoring ${ignore_regex}")
        string_take_regex(str "${ignore_regex}")
    endif()

    string_take(str "${search}")
    ans(match)

    if(NOT match)
      return()
    endif()

    address_set(${rstring} "${str}")

    return_ref(match)
  endfunction()