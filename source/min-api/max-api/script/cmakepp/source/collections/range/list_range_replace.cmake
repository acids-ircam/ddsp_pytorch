
  ## replaces the specified range with the specified arguments
  ## the varags are taken and fill up the range to replace_count
  ## e.g. set(list a b c d e) 
  ## list_range_replace(list "4 0 3:1:-2" 1 2 3 4 5) --> list is equal to  2 4 c 3 1 
  ##
  function(list_range_replace lst_ref range)
    set(lst ${${lst_ref}})

    list(LENGTH lst len)
    range_instanciate(${len} "${range}")
    ans(range)

    set(replaced)
    message("inputlist '${lst}' length : ${len} ")
    message("range: ${range}")
    set(difference)

    range_indices("${len}" ":")
    ans(indices)
    
    range_indices("${len}" "${range}")
    ans(indices_to_replace)
    
    list(LENGTH indices_to_replace replace_count)
    message("indices_to_replace '${indices_to_replace}' count: ${replace_count}")

    math(EXPR replace_count "${replace_count} - 1")

    if(${replace_count} LESS 0)
      message("done\n")
      return()
    endif()

    set(args ${ARGN})
    set(replaced)

    message_indent_push()
    foreach(i RANGE 0 ${replace_count})
      list(GET indices_to_replace ${i} index)

      list_pop_front(args)
      ans(current_value)

      #if(${i} EQUAL ${replace_count})
      #  set(current_value ${args})
      #endif()

      if(${index} GREATER ${len})
        message(FATAL_ERROR "invalid index '${index}' - list is only ${len} long")
      elseif(${index} EQUAL ${len}) 
        message("appending to '${current_value}' to list")
        list(APPEND lst "${current_value}")
      else()
        list(GET lst ${index} val)
        list(APPEND replaced ${val})
        message("replacing '${val}' with '${current_value}' at '${index}'")
        list(INSERT lst ${index} "${current_value}")
        #list(LENGTH current_value current_len)
        math(EXPR index "${index} + 1")
        list(REMOVE_AT lst ${index})
        message("list is now ${lst}")
      endif()



    endforeach()
    message_indent_pop()


    message("lst '${lst}'")
    message("replaced '${replaced}'")
    message("done\n")
    set(${lst_ref} ${lst} PARENT_SCOPE)
    return_ref(replaced)
  endfunction()