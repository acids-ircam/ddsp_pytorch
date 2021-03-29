## `(<length:<int>> <~range...>)-><instanciated range...>`
## 
## instanciates a range.  A uninstanciated range contains anchors
## these are removed when a length is specified (`n`)
## returns a valid range  with no anchors
function(range_instanciate length)
  range_parse(${ARGN})
  ans(range)

  if(${length} LESS 0)
    set(length 0)
  endif()

  math(EXPR last "${length}-1")

  set(result)
  foreach(part ${range})
    string(REPLACE : ";" part ${part})
    set(part ${part})
    list(GET part 0 begin)
    list(GET part 1 end)
    list(GET part 2 increment)
    list(GET part 3 begin_inclusivity)
    list(GET part 4 end_inclusivity)
    list(GET part 5 range_length)
    list(GET part 6 reverse)

    string(REPLACE "n" "${length}" range_length "${range_length}")
    string(REPLACE "$" "${last}" range_length "${range_length}")

    math(EXPR range_length "${range_length}")


    string(REPLACE "n" "${length}" end "${end}")
    string(REPLACE "$" "${last}" end "${end}")

    math(EXPR end "${end}")
    if(${end} LESS 0)
      message(FATAL_ERROR "invalid range end: ${end}")
    endif()

    string(REPLACE "n" "${length}" begin "${begin}")
    string(REPLACE "$" "${last}" begin "${begin}")
    math(EXPR begin "${begin}")
    if(${begin} LESS 0)
      message(FATAL_ERROR "invalid range begin: ${begin}")
    endif()

    list(APPEND result "${begin}:${end}:${increment}:${begin_inclusivity}:${end_inclusivity}:${range_length}:${reverse}")  
  endforeach()
 # message("res ${result}")
  return_ref(result)
endfunction()
