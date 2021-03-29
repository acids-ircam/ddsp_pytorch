
  function(lang target context)    
    #message("target ${target}")
    obj_get(${context} phases)
    ans(phases)

   

    # get target value from
    obj_has(${context} "${target}")
    ans(has_target)
    if(NOT has_target)
      message(FATAL_ERROR "missing target '${target}'")        
    endif()
    obj_get(${context} "${target}")
    ans(current_target)

    if("${current_target}_" STREQUAL "_")
        return()
    endif()

    # check if phase
    list_contains(phases "${current_target}")
    ans(isphase)    
    # if not a phase just return value
    if(NOT isphase)
      return_ref("current_target")
    endif()


    # target is phase 
    map_tryget("${current_target}" name)
    ans(name)


    # get inputs for current target
    obj_get("${current_target}" "input")
    ans(required_inputs)

    # setup required imports
    map_new()
    ans(inputs)
    foreach(input ${required_inputs})
        #message_indent_push()
        #message("getting ${input} ${required_inputs}")

        lang("${input}" ${context})
        ans(res)
        #message("got ${res} for ${input}")
        #message_indent_pop()
        map_set(${inputs} "${input}" "${res}")
    endforeach()

    # handle function call
    map_tryget("${current_target}" function)
    ans(func)
    if("${func}" MATCHES "(.*)\\(([^\\)]*)\\)$" )
        set(func "${CMAKE_MATCH_1}")
        set(arg_assignments "${CMAKE_MATCH_2}")
        string(REPLACE " " ";" arg_assignments "${arg_assignments}")
    else()
        message(FATAL_ERROR "failed to parse targets function")
    endif()

    # curry function to specified arguments
    curry3(() => "${func}"(${arg_assignments}))
    ans(func)

    # compile argument string

    map_keys(${inputs})
    ans(keys)
    set(arguments_string)
    foreach(key ${keys})
      map_tryget(${inputs} "${key}")
      ans(val)
      cmake_string_escape("${val}")
      ans(val)
      #message("key ${key} val ${val}")
      #string(REPLACE "\\" "\\\\"  val "${val}")
      #string(REPLACE "\"" "\\\"" val "${val}")
      set(arguments_string "${arguments_string} \"${val}\"")
    endforeach()
    # call curried function - note that context is available to be modified
    set(func_call "${func}(${arguments_string})")
 
    #message("lang: target '${target}'  func call ${func_call}")
   set_ans("")
    eval("${func_call}")
    ans(res)    
   # message("res '${res}'")
    obj_set(${context} "${target}" "${res}")

    # set single output to return value
    map_tryget(${current_target} output)
    ans(outputs)
    list(LENGTH outputs len)
    if(${len} EQUAL 1)
      set(${context} "${outputs}" "${res}")
    endif()

    map_tryget(${context} "${target}")
    ans(res)

    return_ref(res)
  endfunction()