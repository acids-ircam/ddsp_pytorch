
  macro(encoded_list_remove_at __lst)
    list_remove_at(${__lst} ${ARGN})
  endmacro()