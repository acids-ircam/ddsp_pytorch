## generates a header file from a class definition
  function(cpp_class_header_generate class_def)
    data("${class_def}")
    ans(class_def)
  

    indent_level_push(0)
    set(source)
    string_append_line_indented(source "#pragma once")
    string_append_line_indented(source "")

    cpp_class_generate("${class_def}")
    ans(class_source)
    set(source "${source}${class_source}")


    string_append_line_indented(source "")

    indent_level_pop()
    return_ref(source)
  endfunction()
