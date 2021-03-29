
## quickly extracts string properties values from a json string
## useful for large json files with unique property keys
function(json_extract_string_value key data)
    regex_escaped_string("\"" "\"")
    ans(regex)

    set(key_value_regex "\"${key}\" *: ${regex}")
    string(REGEX MATCHALL "${key_value_regex}" matches "${data}")
    set(values)
    foreach (match ${matches})
        string(REGEX REPLACE "${key_value_regex}" "\\1" match "${match}")
        list(APPEND values "${match}")
    endforeach ()
    return_ref(values)
endfunction()