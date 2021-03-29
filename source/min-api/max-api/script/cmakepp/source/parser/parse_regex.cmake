
  function(parse_regex rstring)
    # deref rstring
    address_get(${rstring})
    ans(str)
   # message("string ${str}")
    # get regex from defintion
    map_get(${definition} regex)
    ans(regex)
   # message("${regex}")

 #   message("parsing '${parser_id}' parser (regex: '${regex}') for '${str}'")
    # try to take regex from string
    
    map_tryget(${definition} ignore_regex)
    ans(ignore_regex)
   # message("ignore: ${ignore_regex}")
    list(LENGTH ignore_regex len)
    if(len)
   # message("ignoring ${ignore_regex}")
        string_take_regex(str "${ignore_regex}")
    endif()
#   message("str is '${str}'")
    string_take_regex(str "${regex}")
    ans(match)

    #message("match ${match}")
    # if not success return
    list(LENGTH match len)
    if(NOT len)
      return()
    endif()
 #   message("matched '${match}'")

    map_tryget(${definition} replace)
    ans(replace)
    if(replace)        
        string_eval("${replace}")
        ans(replace)
        #message("replace ${replace}")
        string(REGEX REPLACE "${regex}" "${replace}" match "${match}")
        #message("replaced :'${match}'")

    endif()

    map_tryget(${definition} transform)
    ans(transform)
    if(transform)
        #message("transforming ")
        call("${transform}"("match"))
        ans(match)
    endif()

    if("${match}_" STREQUAL "_")
        set(match "")
    endif()
    # if success set rstring to rest of string
    address_set(${rstring} "${str}")

    # return matched element
    return_ref(match)
  endfunction()