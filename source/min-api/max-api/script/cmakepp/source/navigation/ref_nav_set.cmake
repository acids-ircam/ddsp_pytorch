## `(<base_value:<any>> ["!"]<navigation expresion> <value...>)-><any>`
##
## sets the specified navigation expression to the the value
## taking into consideration the base_value.
##
##
##
function(ref_nav_set base_value expression)
  string_take(expression "!")
  ans(create_path)

  navigation_expression_parse("${expression}")
  ans(expression)
  set(expression ${expression})

  set(current_value "${base_value}")
  set(current_ranges)
  set(current_property)
  set(current_ref)
  # this loop  navigates through existing values using ranges and properties as navigation expressions
  # the 4 vars declared before this comment will be defined
  while(true)
    list(LENGTH expression continue)
    if(NOT continue)
      break()
    endif()

    list_pop_front(expression)
    ans(current_expression)

    set(is_property true)
    if("${current_expression}" MATCHES "^[<>].*[<>]$")
      set(is_property false)
    endif()
 #   print_vars(current_expression is_property)
    if(is_property)

      #is_map("${current_value}")
      is_address("${current_value}")
      ans(is_ref)
      if(is_ref)
          set(current_ref "${current_value}")
          set(current_property "${current_expression}")
          set(current_ranges) 
      else()
        list_push_front(expression "${current_expression}")
        break()
      endif()

      ref_prop_get("${current_value}" "${current_expression}")
      ans(current_value)
    else()
      list_range_try_get(current_value "${current_expression}")
      ans(current_value)
      list(APPEND current_ranges "${current_expression}")
    endif()
  endwhile()



  set(value ${ARGN})
  
  # if the expressions are left and create_path is not specified
  # this will cause an error else the rest of the path is created
  list(LENGTH expression expression_count)
  if(expression_count GREATER 0)
    if(NOT create_path)
      message(FATAL_ERROR "could not find path ${expression}")
    endif()
    ref_nav_create_path("${expression}" ${value})
    ans(value)
  endif()

  ## get the last existing value
  if(current_ref)
    ref_prop_get("${current_ref}" "${current_property}")
    ans(current_value)
  else()
    set(current_value ${base_value})
  endif()

  ## if there are ranges set the interpret the value as a lsit and set the correct element
  list(LENGTH current_ranges range_count)
  if(range_count GREATER 0)
    list_range_partial_write(current_value "${current_ranges}" "${value}")
  else()
    set(current_value "${value}")
  endif()

  ## either return a new base balue or set the property of the last existing ref
  if(NOT current_ref)    
    set(base_value "${current_value}")
  else()
    ref_prop_set("${current_ref}" "${current_property}" "${current_value}")
  endif()

  return_ref(base_value)
endfunction()


