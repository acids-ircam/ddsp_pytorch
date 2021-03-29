##
##
##
function(interpret_literal token)
 # json_print(${token})
  list(LENGTH token length)
  if(NOT "${length}" EQUAL 1)
    throw("only one token is acceptable, got ${length}" --function interpret_literal)  
  endif()
  map_tryget("${token}" type)
  ans(type)
  if(NOT "${type}" MATCHES "^((unquoted)|(quoted)|(number)|(separated))$")
    throw("invalid type for `literal`: `${type}`" --function interpret_literal)  
  endif()

  map_tryget("${token}" value)
  ans(result_value)
  if("${type}" MATCHES "^((unquoted)|(separated))$")
    if("${type}" STREQUAL "separated")
      string(REGEX MATCH ".(.*)." match "${result_value}")
      set(result_value "${CMAKE_MATCH_1}")
    endif()
    if("${result_value}" MATCHES "^(true)|(false)$")
      set(result_type bool)
    elseif("${result_value}" MATCHES "^0|([1-9][0-9]*)$")
      set(result_type number)
    elseif("${result_value}" STREQUAL "null")
      set(result_type null)
      set(result_value)
    else()
      set(result_type unquoted_string)
    endif()
  elseif("${result_value}" MATCHES "^([\"'])(.*)[\"']$")
    if("${CMAKE_MATCH_1}" STREQUAL "'")
      set(result_type single_quoted_string)
    else()
      set(result_type double_quoted_string)
    endif()
    string(REGEX REPLACE "(\\\\)([\"'])" "\\2" result_value "${CMAKE_MATCH_2}")  
  elseif("${type}" STREQUAL "number")
    set(result_type number)
  endif()

  decode("${result_value}")
  ans(decoded_value)


  cmake_string_escape("${decoded_value}")
  ans(value)

  next_id()
  ans(ref)

  set(code "set(${ref} \"${value}\")\n")


  ast_new(
    "${token}"
    literal             # expression_type
    "${result_type}"    # value_type
    "${ref}"            # ref
    "${code}"           # code
    "${value}"          # value
    "true"              # const
    "true"
    ""                  # children
    )
  ans(ast)

  return(${ast})
endfunction()