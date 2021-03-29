function(interpret_elements tokens separator_type element_types)
    
  
  ## initialize variables
  set(elements)         # stores all single elements
  set(current_tokens)   # set list of current tokens


  ## loop through all tokens 
  ## and collection non-separators inside `current_tokens`
  ## if a separator or `end` is reached parse the `current_tokens`
  ## to obtain an element
  if(tokens)
    list(APPEND tokens end)
    foreach(token ${tokens})
      map_tryget("${token}" type)
      ans(type)

      if("${token}" STREQUAL "end" OR "${type}" MATCHES "^(${separator_type})$")

        interpret_expression_types("${current_tokens}" "${element_types}")
        ans(element)

        if(NOT element)
            throw("failed to interpret element" --function interpret_elements)
        endif() 

        set(current_tokens)
        list(APPEND elements "${element}")

        if("${token}" STREQUAL "end")
          break()
        endif()
      else()
        list(APPEND current_tokens "${token}")
      endif()
    endforeach()
  endif()

  return_ref(elements)  
endfunction()



