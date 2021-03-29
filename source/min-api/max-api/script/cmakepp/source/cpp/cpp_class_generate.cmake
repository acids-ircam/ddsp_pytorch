
  function(cpp_class_generate class_def)
    data("${class_def}")
    ans(class_def)

    map_tryget(${class_def} namespace)
    ans(namespace)


    map_tryget(${class_def} type_name)
    ans(type_name)

    indent_level_push(0)

    set(source)

    string(REPLACE "::" ";" namespace_list "${namespace}")

    foreach(namespace ${namespace_list})
      string_append_line_indented(source "namespace ${namespace}{")
      indent_level_push(+1)
    endforeach()


    string_append_line_indented(source "class ${type_name}{")
    indent_level_push(+1)


    indent_level_pop()
    string_append_line_indented(source "};")


    foreach(namespace ${namespace_list})
      indent_level_pop()
      string_append_line_indented(source "}")
    endforeach()



    indent_level_pop()
    # namespace
    # struct/class
    # inheritance
    # fields
    # methods
    return_ref(source)

  endfunction()
