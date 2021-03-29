## `(<input:<string>>)-><string>`
##
## Transforms the input string to title case.
## Tries to be smart and keeps some words small.
## List of words that are kept small:
## "a, an, and, as, at, but, by, en, for, if, in, of, on, or, the, to, via, vs, v, v., vs."
##
## **Examples**
##  set(input "the function string_totitle works")
##  string_to_title("${input}") # => "The Function string_totitle Works"
##  set(input "testing a small word")
##  string_to_title("${input}") # => "Testing a Small Word"
##
##
function(string_to_title input)
  set(list_keep_small a an and as at but by en for if in of on or the to via vs v v. vs. A An And As At But By En For If In Of On Or The To Via Vs V V. Vs.)
  set(other "[^ ]+")
  set(ws "[ ]+")
  set(is_subsentence true)
  set(res "")

  encoded_list("${input}")
  ans(input_encoded)
  string(REGEX MATCHALL "(${ws})|(${other})" tokens "${input_encoded}")
  
  foreach(token ${tokens})
    if("${token}" MATCHES  "^([^a-zA-Z0-9]*)([a-zA-Z])([a-z]*[']?[a-z]*)([:?!',)]*)$")
      set(pre ${CMAKE_MATCH_1})
      set(first_letter ${CMAKE_MATCH_2})
      set(lc_letters ${CMAKE_MATCH_3})
      set(post ${CMAKE_MATCH_4})
      
      list(FIND list_keep_small "${first_letter}${lc_letters}" index)
      if(index GREATER -1)
         if(is_subsentence)
          string(TOUPPER ${first_letter} first_letter)
        else()
          string(TOLOWER ${first_letter} first_letter)
        endif()
      else()
         string(TOUPPER ${first_letter} first_letter)
      endif()
      
      if("${post}" MATCHES "[^,')]")
        set(is_subsentence true)
      else()
        set(is_subsentence false)
      endif()

      set(token "${pre}${first_letter}${uc_letters}${lc_letters}${post}")
    endif()

    set(res "${res}${token}")
  endforeach()

  return_ref(res)
endfunction()
