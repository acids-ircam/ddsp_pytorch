
## parses and registers a type or returns an existing one by type_name

function(type_def)
  function(type_def)
    data("${ARGN}")
    ans(type)

    if("${type}_" STREQUAL "_")
      set(type any)
    endif()


    list(LENGTH type length)
    if(length GREATER 1)
      map_new()
      ans(t)
      map_set(${t} properties ${type})
      set(type ${t})
    endif()


    is_map("${type}")
    ans(ismap)
    if(ismap)
      map_tryget(${type} type_name)
      ans(type_name)
      if("${type_name}_" STREQUAL "_")
        string(RANDOM type_name)
        map_set("${type}" "anonymous" true)
        #map_set(${type} "type_name" "${type_name}")
      else()
        map_set("${type}" "anonymous" false)
      endif()
    
      map_tryget(data_type_map "${type_name}")
      ans(registered_type)
      if(NOT registered_type)
        map_set(data_type_map "${type_name}" "${type}")
      endif()
      
      map_tryget("${type}" properties)
      ans(props)
      is_map("${props}")
      ans(ismap)
      if(ismap)
        map_iterator("${props}")
        ans(it)
        set(props)
        while(true)
          map_iterator_break(it)
          list(APPEND props "${it.key}:${it.value}")

        endwhile()
        map_set(${type} properties "${props}")
      endif()

      return_ref(type)



    endif()


    map_tryget(data_type_map "${type}")
    ans(res)
    if(res)
      return_ref(res)
    endif()


    map_new()
    ans(res)

    map_set(${res} type_name "${type}")
    map_set(data_type_map "${type}" "${res}")
    return_ref(res)
  endfunction()

  type_def("{
    type_name:'string'
    }")


  type_def("{
    type_name:'int',
    regex:'[0-9]+'
  }")

  type_def("{
    type_name:'any'
  }")


  type_def("{
    type_name:'bool',
    regex:'true|false'
  }")

  
  type_def(${ARGN})
  return_ans()
endfunction()