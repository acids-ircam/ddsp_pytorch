## `(<str:<string>> <length:<int>>)-><first_node:<linked list>>`
##
## Splits the string "str" into multiple parts of length "length". 
## Returns a linked list of the parts
##
## **Examples**
##  set(input "abc")
##  string_split_parts("${input}" 1) # => linked_list("a", "b", "c")
##  string_split_parts("${input}" 2) # => linked_list("ab", "c")
##  string_split_parts("${input}" 3) # => linked_list("abc")
##
##
function(string_split_parts str length)
  address_new()
  ans(first_node)
  if(${length} LESS 1)
    return_ref(first_node)
  endif()
  
  set(current_node ${first_node})
  while(true)      
    string(LENGTH "${str}" len)  

    if(${len} LESS ${length})
      address_set(${current_node} "${str}")
      set(str)
    else()
      string(SUBSTRING "${str}" 0 "${length}" part)
      string(SUBSTRING "${str}" "${length}" -1 str)
      address_set(${current_node} "${part}")
    endif()

    if(str)
      address_new()
      ans(new_node)
      map_set_hidden(${current_node} next ${new_node})
      set(current_node ${new_node})
    else()
      return_ref(first_node)
    endif()     
  endwhile()
endfunction()