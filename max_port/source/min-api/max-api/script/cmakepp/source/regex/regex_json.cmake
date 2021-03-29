macro(regex_json)
  
  if(NOT __regex_json_defined)
    set(__regex_json_defined)
    set(regex_json_string_literal "\"([^\"\\]|([\\][\"])|([\\][\\])|([\\]))*\"")
    set(regex_json_number_literal "[0-9]+")
    set(regex_json_bool_literal "(true)|(false)")
    set(regex_json_null_literal "null")
    set(regex_json_literal "(${regex_json_string_literal})|(${regex_json_number_literal})|${regex_json_bool_literal}|(${regex_json_null_literal})")
  
    set(regex_json_string_token "\"(([\\][\\]\")|(\\\\.)|[^\"\\])*\"")

    set(regex_json_number_token "[0-9\\.eE\\+\\-]+")
    set(regex_json_bool_token "(true)|(false)")
    set(regex_json_null_token "null")
    set(regex_json_object_begin_token "{")
    set(regex_json_object_end_token "}")
    string_codes()
    set(regex_json_array_begin_token "${bracket_open_code}")
    set(regex_json_array_end_token "${bracket_close_code}")
    set(regex_json_separator_token ",")
    set(regex_json_keyvalue_token ":")
    set(regex_json_whitespace_token "[ \t\n\r]+")
    set(regex_json_token "(${regex_json_string_token})|(${regex_json_number_token})|${regex_json_bool_token}|(${regex_json_null_token})|${regex_json_object_begin_token}|${regex_json_object_end_token}|${regex_json_array_begin_token}|${regex_json_array_end_token}|${regex_json_separator_token}|${regex_json_keyvalue_token}|${regex_json_whitespace_token}")


  endif()
endmacro()