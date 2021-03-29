
  function(string_append_line_indented str_ref what)
    indent("${what}" ${ARGN})
    ans(indented)
    set("${str_ref}" "${${str_ref}}${indented}\n" PARENT_SCOPE)
  endfunction()
