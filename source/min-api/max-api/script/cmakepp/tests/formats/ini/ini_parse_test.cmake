function(test)
#http://en.wikipedia.org/wiki/INI_file
  function(ini_parse text)

  endfunction()

  function(ini_parse_lines lines)

  endfunction()

  function(ini_parse_file)


  endfunction()

  # lists are serialized as multi properties
  function(ini_serialize obj)
    set(args ${ARGN})
    list_extract_flag(args --blanklines)
    ans(blanklines)


    obj("${obj}")
    ans(obj)

    map_keys(${obj})
    ans(keys)

    set(ini)

    set(sections)
    foreach(key ${keys})
      map_tryget(${obj} "${key}")
      ans(val)

      is_map("${val}")
      ans(ismap)
      if(NOT ismap)
        foreach(v ${val})
          list(APPEND ini "${key} = ${v}")
        endforeach()  
      else()
        list(APPEND sections "${key}" "${val}")
      endif()
    endforeach()
    list(APPEND ini "")

    foreach(section ${sections})
      is_map("${section}")
      ans(ismap)
      if(ismap)
        map_keys(${section})
        ans(keys)
        foreach(key ${keys})
          map_tryget(${section} "${key}")
          ans(val)
          foreach(v ${val})
            list(APPEND ini "${key} = ${v}")
          endforeach()
        endforeach()
        list(APPEND ini "")
      else()
        list(APPEND ini "[${section}]")
      endif()
    endforeach()


    string(REPLACE ";" "\n" res "${ini}")
    return_ref(res)

  endfunction()

  ini_serialize("{a:1,b:2,c:{a:3,b:4},d:{a:4,b:2}}")
  ans(res)
  message("${res}")

endfunction()