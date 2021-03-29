macro(regex_cmake)
  if(NOT __regex_cmake_included)
    set(__regex_cmake_included true)
  string_codes()

#http://www.cmake.org/cmake/help/v3.0/manual/cmake-language.7.html#grammar-token-regex_cmake_escape_sequence
  
  ## characters
  set(regex_cmake_newline "\n")
  set(regex_cmake_space_chars " \t")
  set(regex_cmake_space "[${regex_cmake_space_chars}]+")
  set(regex_cmake_backslash "\\\\")





  ## tokens

  # line comment
  set(regex_cmake_line_comment "#([^${regex_cmake_newline}]*)")
  set(regex_cmake_line_comment.comment CMAKE_MATCH_1)
  set(regex_cmake_line_comment_no_group "#([^${regex_cmake_newline}]*)")
  
  # bracket_comment
  set(regex_cmake_bracket_comment "#\\[\\[(.*)\\]\\]")
  set(regex_cmake_bracket_comment_no_group "#${bracket_open_code}${bracket_open_code}.*${bracket_close_code}${bracket_close_code}")
 
  # identifier
  set(regex_cmake_identifier "[A-Za-z_][A-Za-z0-9_]*")
  
  # nesting
  set(regex_cmake_nesting_start_char "\\(")
  set(regex_cmake_nesting_end_char "\\)")

  # quoted_argment
  set(regex_quoted_argument "\"([^\"\\]|([\\][\"])|([\\][\\])|([\\]))*\"")
  set(regex_quoted_argument_group "\"(([^\"\\]|([\\][\"])|([\\][\\])|([\\]))*)\"")
  
  # unquoted_argument
  set(regex_unquoted_argument "[^#\\\\\" \t\n\\(\\)]+")



  ## combinations

  # matches every cmake token in a string
  set(regex_cmake_token "(${regex_cmake_bracket_comment_no_group})|(${regex_cmake_line_comment_no_group})|(${regex_quoted_argument})|${regex_unquoted_argument}|${regex_cmake_space}|${regex_cmake_newline}|${regex_cmake_nesting_start_char}|${regex_cmake_nesting_end_char}")


  set(regex_cmake_line_ending "(${regex_cmake_line_comment})?(${regex_cmake_newline})")   
  set(regex_cmake_separation "(${regex_cmake_space})|(${regex_cmake_line_ending})")

  
  ## misc

  # if a value matches this it needs to be put in quotes
  set(regex_cmake_value_needs_quotes "[ \";\\(\\)]")

  set(regex_cmake_value_quote_escape_chars "[\\\\\"]")


  set(regex_cmake_flag "-?-?[A-Za-z_][A-Za-z0-9_\\-]*")
  set(regex_cmake_double_dash_flag "\\-\\-[a-zA-Z0-9][a-zA-Z0-9\\-]*")
  set(regex_cmake_single_dash_flag "\\-[a-zA-Z0-9][a-zA-Z0-9\\-]*")
  
## todo: quoted, unquoated, etc
  set(regex_cmake_argument_string ".*")
  set(regex_cmake_command_invocation "(${regex_cmake_space})*(${regex_cmake_identifier})(${regex_cmake_space})*\\((${regex_cmake_argument_string})\\)")
  set(regex_cmake_command_invocation.regex_cmake_identifier CMAKE_MATCH_2)
  set(regex_cmake_command_invocation.arguments CMAKE_MATCH_4)

  


  set(regex_cmake_function_begin "(^|${cmake_regex_newline})(${regex_cmake_space})?function(${regex_cmake_space})?\\([^\\)\\(]*\\)")
  set(regex_cmake_function_end   "(^|${cmake_regex_newline})(${regex_cmake_space})?endfunction(${regex_cmake_space})?\\(([^\\)\\(]*)\\)")
  set(regex_cmake_function_signature "(^|${cmake_regex_newline})((${regex_cmake_space})?)(${regex_cmake_identifier})((${regex_cmake_space})?)\\([${regex_cmake_space_chars}${regex_cmake_newline}]*(${regex_cmake_identifier})(.*)\\)")
  set(regex_cmake_function_signature.name CMAKE_MATCH_7)
  set(regex_cmake_function_signature.args CMAKE_MATCH_8)
  
 

  endif()
  
endmacro()

