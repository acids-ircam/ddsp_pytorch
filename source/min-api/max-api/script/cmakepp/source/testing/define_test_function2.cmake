
  ## defines a test function
  ## improved version.  use this 
  function(define_test_function2 function_name uut)
    arguments_cmake_string(2 ${ARGC})
    ans(predefined_args)

    #list_extract_flag(predefined_args --timer)
    #ans(timer)
    ## store predefined_args in a ref which is restored 
    ## in function definition
    ## this is necessary because else the args would have to be escaped
    address_new()
    ans(predefined_address)
    address_set(${predefined_address} "${predefined_args}")


    ## define the test function 
    ## which executes the uut and compares
    ## it structurally to the expected value
    define_function("${function_name}" (expected)
      arguments_cmake_string(1 \${ARGC})
      ans(arguments)


      set(scope)
      if("\${expected}" MATCHES "^<(.+)>(.*)$")
        set(scope \${CMAKE_MATCH_1})
        set(expected \${CMAKE_MATCH_2})
      endif()

      obj("\${scope}")
      ans(scope)


      address_get(${predefined_address})
      ans(predefined_args)

      set(__ans)
      set(___code "${uut}(\${predefined_args} \${arguments})")
      
      if(use_timer)
        set(___code "timer_start(t1)\n\${___code}\nans(result)\ntimer_print_elapsed(t1)\nreturn_ref(result)\n")
      endif()
      #print_Vars(___code)
      eval_ref(___code)
      ans(result)

      data("\${expected}")
      ans(expected)

      map_match("\${result}" "\${expected}" )
      ans(res)
      if(NOT res)
        echo_append("actual: ")
        json_print(\${result})
        echo_append("expected: ")
        json_print(\${expected})  
      endif()
      assert(res MESSAGE "values do not match")

      set(test_scope \${scope} PARENT_SCOPE)
      map_keys(\${scope})
      ans(keys)
      foreach(key \${keys})
        map_tryget(\${scope} "\${key}")
        ans(current_value)

        map_match("\${\${key}}" "\${current_value}")
        ans(success)

        assert(success MESSAGE "scope var '\${key}' (\${\${key}}) does not match expected value '\${current_value}'")
      endforeach()

      return_ref(result)
    )
    return()
  endfunction()
