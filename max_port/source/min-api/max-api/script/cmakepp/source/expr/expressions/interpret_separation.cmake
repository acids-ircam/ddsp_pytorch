function(interpret_separation tokens separator_type separator_char pre_element post_element)

    
    ## initialize variables
    set(elements)         # stores all single elements
    set(current_tokens)   # set list of current tokens
    set(const true)      # set if all elements are const
    set(argument)         # set derived argument
    set(code)             # set derived code


    ## loop through all tokens 
    ## and collection non-separators inside `current_tokens`
    ## if a separator or `end` is reached parse the `current_tokens`
    ## to obtain an element
    list(APPEND tokens end)
    foreach(token ${tokens})
      map_tryget("${token}" type)
      ans(type)
     # print_vars(type)
      if("${token}" STREQUAL "end" OR "${type}" MATCHES "^(${separator_type})$")

        interpret_expression("${current_tokens}" ${ARGN})
        ans(element)

        if(NOT element)
            throw("failed to interpret element")
        endif() 
        set(current_tokens)
        list(APPEND elements "${element}")

        map_tryget("${element}" const)
        ans(is_const)

        if(NOT is_const)
          set(const false)
        endif()

        map_tryget("${element}" code)
        ans(element_code)

        map_tryget("${element}" argument)
        ans(element_argument)

        set(code "${code}${element_code}")
        set(argument "${argument}${separator_char}${pre_element}${element_argument}${post_element}")

        if("${token}" STREQUAL "end")
          break()
        endif()
      else()
        list(APPEND current_tokens "${token}")
      endif()
    endforeach()
    string(LENGTH argument argument_length)
    if(argument_length)
        # remove leading whitespace
        string(SUBSTRING "${argument}" 1 -1 argument)
    endif()
    map_new()
    ans(ast)
    map_set("${ast}" type separation)
    map_set("${ast}" elements "${elements}")
    map_set("${ast}" code "${code}")
    map_set("${ast}" argument "${argument}")
    map_set("${ast}" const "${const}")
    return(${ast})
endfunction()


