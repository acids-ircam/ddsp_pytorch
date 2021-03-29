

# Converts a CMake list to a string containing elements separated by spaces
function(list_to_string  list_name separator )
  set(res)
  set(current_separator)
  foreach(element ${${list_name}})
    set(res "${res}${current_separator}${element}")
    # after first iteration separator will be set correctly
    # so i do not need to remove initial separator afterwords
    set(current_separator ${separator})
  endforeach()
  return_ref(res)

endfunction()