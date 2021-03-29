## `(<~range...>)-><range>`
##
## parses a range string and normalizes it to have the following form:
## `<range> ::= <begin>":"<end>":"<increment>":"<begin inclusivity:<bool>>":"<end inclusivity:<bool>>":"<length>":"<reverse:<bool>>
## these `<range>`s can be used to generate a index list which can in turn be used to address lists.
##  
##   * a list of `<range>`s is a  `<range>`  
##   * `$` the last element 
##   * `n` the element after the last element ($+1)
##   * `-<n>` a begin or end starting with `-` is transformed into `$-<n>`
##   * `"["` `"("` `")"` and `"]"`  signify the inclusivity.  
## 
function(range_parse)
  ## normalize input by replacing certain characters
  string(REPLACE " " ";" range "${ARGN}")
  string(REPLACE "," ";" range "${range}")

  string(REPLACE "(" ">" range "${range}")
  string(REPLACE ")" "<" range "${range}")
  string(REPLACE "[" "<" range "${range}")
  string(REPLACE "]" ">" range "${range}")

  ## if there is more than one range group 
  ## recursively invoke range_parse
  list(LENGTH range group_count)
  set(ranges)
  if(${group_count} GREATER 1)
    foreach(group ${range})
      range_parse("${group}")
      ans(current)
      list(APPEND ranges "${current}")
    endforeach()
    return_ref(ranges)
  endif()


  ## get begin and end_inclusivity chars
  ## results in begin_inclusivity and end_inclusivity to be either "<" ">" or " "
  string(REGEX REPLACE "([^<>])+" "_" inclusivity "${range}")
  set(inclusivity "${inclusivity}___")
  string(SUBSTRING ${inclusivity} 0 1 begin_inclusivity )
  string(SUBSTRING ${inclusivity} 1 1 end_inclusivity )
  string(SUBSTRING ${inclusivity} 2 1 three )
  if(${end_inclusivity} STREQUAL _)
    set(end_inclusivity ${three})
  endif()


  ## transform "<" ">" and " " to a true or false value
  ## " " means default inclusivity
  set(default_begin_inclusivity)
  set(default_end_inclusivity)

  if("${begin_inclusivity}" STREQUAL "<")
    set(begin_inclusivity true)
  elseif("${begin_inclusivity}" STREQUAL ">")
    set(begin_inclusivity false)
  else()
   set(begin_inclusivity true)
   set(default_begin_inclusivity true) 
  endif()

  if("${end_inclusivity}" STREQUAL "<")
    set(end_inclusivity false)
  elseif("${end_inclusivity}" STREQUAL ">")
    set(end_inclusivity true)
  else()
    set(end_inclusivity true)
    set(default_end_inclusivity true)
  endif()

  ## remove all angular brackets from current range
  string(REGEX REPLACE "[<>]" "" range "${range}")

  ## default range for emtpy range (n:n)
  if("${range}_" STREQUAL "_")
    set(range "n:n:1")
    if(default_end_inclusivity)
      set(end_inclusivity false)
    endif()
  endif()

  ## default range for * 0:n
  if("${range}" STREQUAL "*")
    set(range "0:n:1")
  endif()

  ##  default range for  : 0:$
  if("${range}" STREQUAL ":")
    set(range "0:$:1")
  endif()

  ## split list at ":"
  string(REPLACE  ":" ";" range "${range}")
  
  ## normalize range and simplify elements
  

  ## single number is transformed to i;i;1 
  list(LENGTH range part_count)
  if(${part_count} EQUAL 1)
    set(range ${range} ${range} 1)
  endif()

  ## 2 numbers is autocompleted to  i;j;1
  if(${part_count} EQUAL 2)
    list(APPEND range 1)
  endif()

  ## now every range has 3 number begin end and increment
  list(GET range 0 begin)
  list(GET range 1 end)
  list(GET range 2 increment)

  ## if part count is higher than 3 the begin_inclusivity is specified
  if(${part_count} GREATER 3)
    list(GET range 3 begin_inclusivity)
  endif()
  ## if part count is higher than 4 the end_inclusivity is specified
  if(${part_count} GREATER 4)
    list(GET range 4 end_inclusivity)
  endif()

  ## invalid range end must be reachable from end using the specified increment
  if((${end} LESS ${begin} AND ${increment} GREATER 0) OR (${end} GREATER ${begin} AND ${increment} LESS 0))
    return()
  endif()

  ## set wether the range is reverse or forward
  set(reverse false)
  if(${begin} GREATER ${end})
    set(reverse true)
  endif()

  ## some special cases  -0 = $ (end)
  if(${begin} STREQUAL -0)
    set(begin $)
  endif()
  if(${end} STREQUAL -0)
    set(end $)
  endif()

  ## create math expression to calculate begin and end if anchors are used
  ## negative begin or end is transformed into $-i 
  set(begin_negative false)
  set(end_negative false)
  if(${begin} LESS 0)
    set(begin "($${begin})")
    set(begin_negative true)
  endif()
  if(${end} LESS 0)
    set(end "($${end})")
    set(end_negative true)
  endif()

  ## if begin or end contains a sign operator
  ## put it in parentheses
  if("${begin}" MATCHES "[\\-\\+]")
    set(begin "(${begin})")
  endif()
  if("${end}" MATCHES "[\\-\\+]")
    set(end "(${end})")
  endif()

  ## calculate length of range (number of elements that are spanned)
  ## depending on the orientation of the range 
  if(NOT reverse)
    set(length "${end}-${begin}")
    if(end_inclusivity)
      set(length "${length}+1")
    endif()
    if(NOT begin_inclusivity)
      set(length "${length}-1")
    endif()
  else()
    set(length "${begin}-${end}")
    if(begin_inclusivity)
      set(length "${length}+1")
    endif()
    if(NOT end_inclusivity)
      set(length "${length}-1")
    endif()
  endif()

  ## simplify some typical ranges 
  string(REPLACE "n-n" "0" length "${length}")
  string(REPLACE "n-$" "1" length "${length}")
  string(REPLACE "$-n" "0-1" length "${length}")
  string(REPLACE "$-$" "0" length "${length}")

  ## recalculate length by dividing by step size
  if("${increment}" GREATER 1)
    set(length "(${length}-1)/${increment}+1")
  elseif("${increment}" LESS -1)
    set(length "(${length}-1)/(0-(0${increment}))+1")
  elseif(${increment} EQUAL 0)
    set(length 1)
  endif()

  ## if no anchor is used the length can be directly computed
  if(NOT "${length}" MATCHES "\\$|n" )
    math(EXPR length "${length}")
  else()
     # 
  endif()

  ## set the range string and return it
  set(range "${begin}:${end}:${increment}:${begin_inclusivity}:${end_inclusivity}:${length}:${reverse}")

  return_ref(range)
endfunction()