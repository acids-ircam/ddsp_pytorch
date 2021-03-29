  #extracts a single flag from a list returning true if it was found
  # false otherwise. 
  # if flag exists multiple time online the first instance of the flag is removed
  # from the list
 function(list_extract_flag __list_extract_flag flag)
    list(FIND "${__list_extract_flag}" "${flag}" idx)
    if(${idx} LESS 0)
      return(false)     
    endif()
    list(REMOVE_AT "${__list_extract_flag}" "${idx}") 
    set("${__list_extract_flag}" "${${__list_extract_flag}}" PARENT_SCOPE)
    return(true)
endfunction()

