
  function(list_structure_print_help structure)
    map_keys(${structure} )
    ans(keys)

    set(descriptors)
    set(structure_help)
    foreach(key ${keys})

      map_get(${structure}  ${key})
      ans(descriptor)
      value_descriptor_parse(${key} ${descriptor})
      ans(descriptor)
      list(APPEND descriptors ${descriptor})

      scope_import_map(${descriptor})
      set(current_help)
      list(GET labels 0 first_label)
      set(current_help ${first_label})

      if(NOT "${default}_" STREQUAL "_")
        set(current_help "[${current_help} = ${default}]")
      elseif(${min} EQUAL 0 )
        set(current_help "[${current_help}]")
      endif()


      set(structure_help "${structure_help} ${current_help}")

    endforeach()
    if(structure_help)
      string(SUBSTRING "${structure_help}" 1 -1 structure_help)
    endif()
    message("${structure_help}")
    message("Details: ")
    foreach(descriptor ${descriptors})
      scope_import_map(${descriptor})
      list_to_string( labels ", ")
      ans(res)
      message("${displayName}: ${res}")
      if(description)
        message_indent_push()
        message("${description}")
        message_indent_pop()
      endif()

    endforeach()
  endfunction()
