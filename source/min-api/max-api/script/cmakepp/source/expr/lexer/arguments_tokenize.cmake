
macro(arguments_tokenize first last)

  math(EXPR _last "${last} - 1")
  #_message("first ${first} last ${last} _last ${_last}")
  if(NOT ${first} GREATER ${_last})
    set(input)
    foreach(i RANGE ${first} ${_last})
      set(input "${input}${ARGV${i}}")
    endforeach()
    string(REPLACE ";" "" input "${input}")
    string(REPLACE "(" "" input "${input}")
    string(REPLACE ")" "" input "${input}")
    string(REPLACE "[" "" input "${input}")
    string(REPLACE "]" "" input "${input}")

   # _message("input '${input}'")
    ## tokenize the string
   # string(REPLACE "" ";;" input "${input}")
    set(token_chars "{},:=&\\*\\$\\.\\-\\+\\|\\^%#@!\\?\\/~<>")

    set(token_regex "([^\"';${token_chars}]*)|||([\"']([^\"'\\]|([\\]['\"])|([\\][\\])|([\\]))*[\"'])|('.?')|[ \t]+||||||{|}|,|:|=|&|\\*|\\$|\\.|\\-|\\+|\\||\\^|%|#|@|!|\\?|\\/|~|\n|<|>|([^ \t{},:=&\\*\\$\\.\\-\\+\\|\\^%#@!\\?\\/~<>\n]+)")
    
    
    string(REGEX MATCHALL "${token_regex}" token_strings "${input}")
    string(REGEX REPLACE "${token_regex}" "" error "${input}")
   # messaGE("${token_strings}")
    if(NOT error)

      set(type_list 
        invalid         # 0
        number          # 1
        quoted          # 2
        char            # 3
        bracket_open    # 4
        bracket_close   # 5
        paren_open      # 6
        paren_close     # 7
        semicolon       # 8 
        brace_open      # 9
        brace_close     # 10
        comma           # 11
        colon           # 12
        equals          # 13
        ampersand       # 14
        asterisk        # 15
        dollar          # 16
        dot             # 17
        minus           # 18
        plus            # 19
        pipe            # 20
        zirkumflex      # 21
        modulo          # 22
        hash            # 23
        at              # 24
        exclamation_mark# 25
        question_mark   # 26
        slash           # 27
        tilde           # 28
        new_line        # 29
        angular_open    # 30
        angular_close   # 31
        separated       # 32
        separation_open # 33
        separation_close# 34
        white_space     # 35
        unquoted        # 36
        )
      
      ## replace the token_strings with their respective indices in the type_list
      ## use the index indicator  so that normal numbers are not confused
      ## with token indices 
      set(token_codes ${token_strings})
      string(REGEX REPLACE  ";[0-9]+;" ";1;"  token_codes "${token_codes}" )
      string(REGEX REPLACE "[\"']([^\"'\\]|([\\][\"'])|([\\][\\])|([\\]))*[\"']" "2" token_codes "${token_codes}")  
      string(REGEX REPLACE "'.?'" "3" token_codes "${token_codes}")
      string(REGEX REPLACE "" "4" token_codes "${token_codes}")
      string(REGEX REPLACE "" "5" token_codes "${token_codes}")
      string(REGEX REPLACE "" "6" token_codes "${token_codes}")
      string(REGEX REPLACE "" "7" token_codes "${token_codes}")
      string(REGEX REPLACE "" "8" token_codes "${token_codes}")
      string(REGEX REPLACE "{" "9" token_codes "${token_codes}")
      string(REGEX REPLACE "}" "10" token_codes "${token_codes}")
      string(REGEX REPLACE "," "11" token_codes "${token_codes}")
      string(REGEX REPLACE ":" "12" token_codes "${token_codes}")
      string(REGEX REPLACE "=" "13" token_codes "${token_codes}")
      string(REGEX REPLACE "&" "14" token_codes "${token_codes}")
      string(REGEX REPLACE "\\*" "15" token_codes "${token_codes}")
      string(REGEX REPLACE "\\$" "16" token_codes "${token_codes}")
      string(REGEX REPLACE "\\." "17" token_codes "${token_codes}")
      string(REGEX REPLACE "\\-" "18" token_codes "${token_codes}")
      string(REGEX REPLACE "\\+" "19" token_codes "${token_codes}")
      string(REGEX REPLACE "\\|" "20" token_codes "${token_codes}")
      string(REGEX REPLACE "\\^" "21" token_codes "${token_codes}")
      string(REGEX REPLACE "%" "22" token_codes "${token_codes}")
      string(REGEX REPLACE "#" "23" token_codes "${token_codes}")
      string(REGEX REPLACE "@" "24" token_codes "${token_codes}")
      string(REGEX REPLACE "!" "25" token_codes "${token_codes}")
      string(REGEX REPLACE "\\?" "26" token_codes "${token_codes}")
      string(REGEX REPLACE "\\/" "27" token_codes "${token_codes}")
      string(REGEX REPLACE "~" "28" token_codes "${token_codes}")
      string(REGEX REPLACE "\n" "29" token_codes "${token_codes}")
      string(REGEX REPLACE "<" "30" token_codes "${token_codes}")
      string(REGEX REPLACE ">" "31" token_codes "${token_codes}")
      string(REGEX REPLACE "[^;${token_chars}]*" "32" token_codes "${token_codes}")
      string(REGEX REPLACE "" "33" token_codes "${token_codes}")
      string(REGEX REPLACE "" "34" token_codes "${token_codes}")
      string(REGEX REPLACE "[ \t]+" "35" token_codes "${token_codes}")
      string(REGEX REPLACE ";[^][^;]*;" ";36;" token_codes "${token_codes}" )

      ## remove token type index indicator to obtain pure indices
      string(REPLACE "" "" token_codes "${token_codes}")

      if(token_codes)
        list(GET type_list ${token_codes} token_types)
      else()
        set(token_types)
      endif()
      list(LENGTH token_strings token_count)
    else()
      set(error "could not tokenize '${error}'")
    endif()
  else()
    set(error invalid_input_args)
  endif()
endmacro()
