function(test)


  cmakepp_config(base_dir)
  ans(base_dir)



  set(func_string "



##asdasdasd
    ## comment


##asdasdasd
    ## comment


##asdasdasd
    ## comment


##asdasdasd
    ## comment


##asdasdasd
    ## comment


##asdasdasd
    ## comment


##asdasdasd
    ## comment


##asdasdasd
    ## comment


##asdasdasd
    ## comment


##asdasdasd
    ## comment



##asdasdasd
    ## comment

    function(  __as23d
        dasd
      )

    function(ads)

    endfunction()
        function(bsd

          )

        endfunction()
    endfunction(
      )


    function(csd)

    endfunction(

      )

    ")



    function(cmake_function_instanciate code name) 
      cmake_function_rename_first("${code}" "${name}")
      ans(res)
      eval("${res}")
    endfunction()



    timer_start(t1)
    function_string_rename("${func_string}" "kkasasd")
    ans(res)
    timer_print_elapsed(t1)

    timer_start(t1)
    cmake_function_rename_first("${func_string}" kkasasd)
    ans(res)
    timer_print_elapsed(t1)

    function(function_instanciate code name)
      if(COMMAND "${code}")
        defined_function_instanciate("${code}" "${name}")
        return_ans()
      endif()

      is_lambda("${code}")
      ans(is_lambda)
      if(is_lambda)
        lambda2_instanciate("${code}" "${name}")
        return_ans()
      endif()

      is_cmake_function("${code}")
      ans(is_cmake_function)
      if(is_cmake_function)
        cmake_function_instanciate("${code}" "${name}")
        return_ans()
      endif()

      message(FATAL_ERROR "'${code}' is not a function")

    endfunction()

    function(defined_function_instanciate code name)
      if("${code}" STREQUAL "${name}")
        return()
      endif()
      eval("
        function(${name} )
          ${code}(\${ARGN})
          set(__ans \${__ans} PARENT_SCOPE)
        endfunction()
      ")
    endfunction()



  function(function_import2 callable)
    set(args ${ARGN})
    list_extract_flag(args --redefine)
    ans(redefine)
    list_extract_labelled_value(args "=>")
    ans(function_name)

    if(NOT callable)
      message(FATAL_ERROR "no callable specified")
    endif()

    ## nothing needs to be done if callable is a command
    ## and no function name was specified or function name is the same as callable
    if(COMMAND "${callable}")
      if(NOT function_name OR "${callable}_" STREQUAL "${function_name}_")
        return_ref(callable)
      endif()
    endif()



    if(NOT function_name)
      function_new()
      ans(function_name)
    else()
      if(COMMAND "${function_name}" AND NOT redefine)
        messagE(FATAL_ERROR "cannot import '${callable}' as '${function_name}' because it already exists")
      endif()
    endif()


    function_instanciate("${callable}" "${function_name}")
    ans(compiled)
    return_ref(function_name)

  endfunction()


  function_import2("[]()return(3)" => asd_fu)
  asd_fu()
  ans(res)
  assert("${res}" EQUAL 3)

  function_import2("function(__)\nreturn(4)\nendfunction()" => asd_fu --redefine)
  asd_fu()
  ans(res)
  assert(${res} EQUAL 4)

  function_import2(asd_fu => bsd_fu)
  bsd_fu()
  ans(res)
  assert(${res} EQUAL 4)

  timer_start(t1)
  foreach(i RANGE 0 50)
    #function_import2("[]()return(3)" => asd_fu --redefine)
    #function_import2("function(__)\nreturn(4)\nendfunction()" => asd_fu --redefine)
    function_import2(asd_fu => bsd_fu --redefine)
  endforeach()
  timer_print_elapsed(t1)


  timer_start(t1)
  foreach(i RANGE 0 50)
    #function_import("[]()return(3)" as asd_fu REDEFINE)
    #function_import("function(__)\nreturn(4)\nendfunction()" as asd_fu REDEFINE)
    function_import(asd_fu as bsd_fu REDEFINE)
  endforeach()
  timer_print_elapsed(t1)
endfunction()