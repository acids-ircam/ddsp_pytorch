## declares a programmou able property 
## if one var arg is specified the function is ussed as a getter
## if there are more the one args you need to label the getter with --getter and setter with --setter
## if no var arg is specified the two functions will be created call
## get_${property_name} and set_${property_name}

  function(obj_declare_property obj property_name)
    set(args ${ARGN})
    list_extract_flag(args --hidden)
    ans(hidden)
    if(hidden)
      set(hidden --hidden)
    else()
      set(hidden)
    endif()

    list(LENGTH args len)
    if(${len} EQUAL 0)
      set(getter "get_${property_name}")
      set(setter "set_${property_name}")
    elseif(${len} GREATER 1)
      list_extract_labelled_value(args --getter)
      ans(getter)
      list_extract_labelled_value(args --setter)
      ans(setter)
    else()
      set(getter ${args})
    endif()

    if(getter)
      obj_declare_property_getter("${obj}" "${property_name}" "${getter}" ${hidden})
      set(${getter} ${${getter}} PARENT_SCOPE)
    endif()
    if(setter)
      obj_declare_property_setter("${obj}" "${property_name}" "${setter}" ${hidden})
      set(${setter} ${${setter}} PARENT_SCOPE)
    endif()
  endfunction()
