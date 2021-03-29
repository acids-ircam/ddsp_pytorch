## `(<cmake code> [--extended])-><cmake token>...`
##
## this function parses cmake code and returns a list linked list of tokens 
##
## ```
## <token> ::= { 
##  type: "command_invocation"|"bracket_comment"|"line_comment"|"quoted_argument"|"unquoted_argument"|"nesting"|"nesting_end"|"file"
##  value: <string> the actual string as is in the source code 
##  [literal_value : <string>] # the value which actually is meant (e.g. "asd" -> asd  | # I am A comment -> ' I am A comment')
##  next: <token>
##  previous: <token>
## }
## <nesting token> ::= <token> v {
##   "begin"|"end": <nesting token>
## }
## <extended token> ::= (<token>|<nesting token>) v {
##  line:<uint> # the line in which the token is found
##  column: <uint> # the column in which the token starts
##  length: <uint> # the length of the token 
## }
## ```
function(cmake_tokens_parse code)
  set(args ${ARGN})
  list_extract_flag(args --extended)
  ans(extended)

  regex_cmake()

  set(line_counter 1)
  set(column_counter 1)
  set(nestings)
  set(previous)
  set(tokens)

  

  ## encode list to remove unwanted codes
  string_encode_list("${code}") # string replace \r ""?
  ans(code)
  string(REGEX MATCHALL "${regex_cmake_token}" literal_values "${code}")
  # this code checks for errors... however it is slow
  #string(REGEX REPLACE "${regex_cmake_token}" "" error "${code}")
  #if(error)
  #  throw("could not parse '{error}'" --function cmake_tokens_parse)
  #endif()
  
  while(true)
    list(LENGTH literal_values literals_left)
    if(NOT literals_left)
      break()
    endif()

    list(GET literal_values 0 literal)
    list(REMOVE_AT literal_values 0)

    map_new()
    ans(token)
    list(APPEND tokens ${token})

    set(literal_value "${literal}")
    if("${literal}_" STREQUAL "(_")
      set(type nesting)
      list(INSERT nestings 0 ${token})

    elseif("${literal}_" STREQUAL ")_")
      set(type nesting_end)
      if(NOT nestings)
        throw("unbalanced nesting expressions" --function cmake_tokens_parse)
      endif()
      list(GET nestings 0 begin)
      list(REMOVE_AT nestings 0)
      map_set(${begin} end ${token})
      map_set(${token} begin ${begin})

    elseif("${literal}_" MATCHES "^${regex_cmake_space}_$")
      set(type white_space)
    elseif("${literal}_" STREQUAL "${regex_cmake_newline}_")
      set(type new_line)
    else()
      ## all literals here need the decoded literal value
      string_decode_list("${literal}")
      ans(literal)
      if("${literal}_" MATCHES "^${regex_cmake_bracket_comment}_$")
        set(type bracket_comment)
        set(literal_value "${CMAKE_MATCH_1}")
      elseif("${literal}_" MATCHES "^${regex_cmake_line_comment}_$")
        set(type line_comment)
        set(literal_value "${CMAKE_MATCH_1}")
      elseif("${literal}_" MATCHES "^${regex_unquoted_argument}_$")
        set(type unquoted_argument)
        cmake_string_unescape("${literal}")
        ans(literal_value)
        set(literal_value "${literal_value}")
      
        if(NOT nestings AND "${literal}_" MATCHES "^${regex_cmake_identifier}_$")
          if("_${literal_values}" MATCHES "^_(${regex_cmake_space};)?${regex_cmake_nesting_start_char};")
            set(type command_invocation)
          endif()
        endif()
      elseif("${literal}_" MATCHES "^\"(.*)\"_$")
        set(type quoted_argument)
        cmake_string_unescape("${CMAKE_MATCH_1}")
        ans(literal_value)   
        set(literal_value "${literal_value}")
      else()
        throw("unknown token '{literal}'")
      endif()
    endif()

    map_set(${token} type "${type}")
    map_set(${token} value "${literal}")
    map_set(${token} literal_value "${literal_value}")
    
    if(extended) #these are computed values which make parsing slow
      if(nestings)
        list(GET nestings 0 current_nesting)
        map_append(${current_nesting} children ${token})
        map_set(${token} parent "${current_nesting}")
      endif()

      map_set_hidden(${token} previous ${previous})
      map_property_string_length("${token}" value)
      ans(length)
      map_set("${token}" length "${length}")
      map_set(${token} line ${line_counter})
      map_set(${token} column ${column_counter})
      math(EXPR  column_counter "${column_counter} + ${length}")
      if("${type}" STREQUAL "new_line")
        set(column_counter 0)
        math(EXPR line_counter "${line_counter} + 1")
      endif()
    endif()

    ## setup the linked list
    if(previous)
      map_set_hidden(${previous} next ${token})
    endif()
    set(previous ${token})
    list(APPEND tokens ${token})
  endwhile()
  cmake_token_eof()
  ans(eof)
  if(previous)
    map_set(${previous} next ${eof})
    map_set(${eof} previous ${previous})
  endif()

  list(APPEND tokens ${eof})
    
  return_ref(tokens)
endfunction() 

