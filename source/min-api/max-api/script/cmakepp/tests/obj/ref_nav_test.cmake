function(test)


  function(refset values expressions force value)
    list_pop_front(expressions)
    ans(current_expression)

    set(result)

    print_vars(expression indices expressions force value refs )

    if("${current_expression}" MATCHES "^[<>].*[<>]$")
      message("range")
    else()
    foreach(ref ${refs})
      is_address("${ref}")
      ans(is_address)
      if(is_address)
        list(APPEND result "${ref}")
      else()
        if(force)
          map_new()
          ans_apend(result)
        endif()
      endif()
    endforeach()


    foreach(ref ${result})
      #refset("${")
      map_set("${ref}" "${property}" "${value}")
    endforeach()


    endif()
    

    return()
  endfunction()


  data("[{a:1},{a:2}]")
  ans(dat)

  navigation_expression_parse("a")  
  ans(expression)

  refset("${dat}" "" "" "${expression}" false "myval")
  ans(res)



  json_print(${res})






  return()

  range_indices(-1 "[10:5:-1]")
  ans(indices)

  _message("${indices}")
  


  set(list "")

  if(DEFINED list)
    message("is defined")
  else()
    message("undefined")

  endif()
  list_append(list "")

  list(LENGTH list len)

  _message("list [${len}] ${list}")
  foreach(l ${list})
    _message("element- '${i}'")
  endforeach()

  return()


  function(ref_nav base_value expression)

  navigation_expression_parse("${expression}")
  ans(expression)

  set(current_value "${base_value}")
  set(current_ranges)
  set(current_property)
  set(current_ref)
  set(current_indices)

  set(force false)

  # this loop  navigates through existing values using ranges and properties as navigation expressions
  # the 4 vars declared before this comment will be defined
  while(true)

    list(LENGTH expression continue)
    if(NOT continue)
      break()
    endif()


    list_pop_front(expression)
    ans(current_expression)

    if(NOT force)
      string_take(current_expression "!")
      ans(force)
      if(force)
        set(force true)      
      endif()
    endif()


    if("${current_expression}" MATCHES "^[<>].*[<>]$")
      # handle ranges/ predicates / ...
      list(LENGTH current_value __len)
      list_range_try_get(current_value "${current_expression}")
      ans(current_value)
      range_indices("${__len}" "${current_expression}")
      ans(current_indices)
      list(APPEND current_ranges "${current_expression}")
    else()
      # handle properties
      set(tmp ${current_value})
      set(current_value)
      set(tmp2 ${current_ref})
      set(current_ref)

      foreach(value ${tmp})
        is_address("${value}")
        ans(is_ref)
        if(is_ref)
            set(current_ref ${current_ref} "${value}")
            set(current_property "${current_expression}")
            set(current_ranges) 
            set(current_indices)
        endif()
  
        ref_prop_get("${value}" "${current_expression}")
        ans_append(current_value)
  


      endforeach()

      if(NOT current_ref)
        set(current_ref ${tmp2})
      endif()   
    
    endif()
    print_vars(current_expression expression current_value current_ranges current_property current_indices current_ref force)
  endwhile()


return()
endfunction()





  function(test_ref_nav current_value expression)
    data("${current_value}")
    ans(current_value)

    ref_nav("${current_value}" "${expression}")
    return_ans()
  endfunction()

  define_test_function(test_uut test_ref_nav current_value expression)

  test_ref_nav("" "[1:2].!a" "hello")

 # test_ref_nav("[1,2,3,4,5]" "[9:10]")
  #test_ref_nav("[{a:[{},1,{b:3}]},2,{a:{b:1}}]" "[:].a")
 # test_uut("" "[{a:1},{a:{b:1}]" "[1[" 234)

endfunction()