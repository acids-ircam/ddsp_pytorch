# `(<input:<string>>)-><cmake code>`
#
#
# creates and returns cmake code from specified template string
# the syntax is as follows
# * `<%%` `%%>` encloses cmake code
# * `<%%%` and `%%%>` escape `<%%` and `%%>` resp.
# * shorthands
#     * `<%%=` runs the function specified if possible (only single line function calls allowed) or formats the following nonspace string using the `format()` function (allows things like `<%%="${var} {format.expression[3].navigation.key}%%>`)
#     * single line function calls are `func(.....)` but not `func(... \n ....)`
#     * `@@<cmake function call>` is replaced with `<%%= <cmake function call> %%>`
#     * `@@<navigation expression>` is replaced with `<%%= {<navigation expression>}%%>`
#
# **Examples:**
# * assume `add(lhs rhs) => {lhs+rhs}`
# * assume `data = {a:1,b:[2,3,4],c:{d:5,b:[6,7,8]}}`
# * assume `data2 = "hello!"`
# * `@@@@` => `@@`
# * `<%%%` => `<%%`
# * `%%%>` => `%%>`
# * `@@data2` => `hello!`
# * `@@add(1 4)` => `5`
# * `@@foreach(i RANGE 1 3)@@i@@endforeach()` => `123`
# * `<%%= ${data2} %%>` => `hello!`
# * `<%%= ${data2} ${data2} bye! %%>` => `hello!;hello!;bye!`
# * `<%%= "${data2} ${data2} bye!" %%>` => `hello! hello! bye!`
# * `<%%= add(1 4) %%> => `5`
# * `<%% template_out(hi) %%>` => `hi`
#
# **NOTE** *never use ascii 16 17 18 28 29 31* as these special chars are used internally
function(template_compile input)
  regex_cmake()
  # encode input
  set(delimiter_start "<%")
  set(delimiter_end "%>")
  set(delimiter_start_escape "<%%")
  set(delimiter_end_escape "%%>")
  set(shorthand_indicator "@")
  set(shorthand_indicator_escape "@@")

  string(ASCII 16 shorthand_indicator_code)
  string(ASCII 17 delimiter_code)
  string(ASCII 18 delimiter_start_escape_code)
  string(ASCII 19 delimiter_end_escape_code)
  string(ASCII 20 shorthand_indicator_escape_code)

  string(REPLACE "${shorthand_indicator_escape}"
                 "${shorthand_indicator_escape_code}" input "${input}")
  string(REPLACE "${delimiter_start_escape}" "${delimiter_start_escape_code}"
                 input "${input}")
  string(REPLACE "${delimiter_end_escape}" "${delimiter_end_escape_code}" input
                 "${input}")

  string_encode_semicolon("${input}")
  ans(input)
  string_encode_bracket("${input}")
  ans(input)
  string(REPLACE "${delimiter_start}" "${delimiter_code}" input "${input}")
  string(REPLACE "${delimiter_end}" "${delimiter_code}" input "${input}")
  string(REPLACE "${shorthand_indicator}" "${shorthand_indicator_code}" input
                 "${input}")

  # match all fragments (literal and code fragments)
  set(code_fragment_regex
      "${delimiter_code}([^${delimiter_code}]*)${delimiter_code}")
  set(literal_fragment_regex "([^${delimiter_code}]+)")
  set(regex_cmake_function
      "(${regex_cmake_identifier})?\\(([^\n${shorthand_indicator_code}]*)\\)")

  string(REGEX
         REPLACE "${shorthand_indicator_code}(${regex_cmake_function})"
                 "${delimiter_code}=\\1${delimiter_code}" input "${input}")

  string(
    REGEX
    REPLACE
      # "${shorthand_indicator_code}([^
      # ${shorthand_indicator_code}${delimiter_code}\"'\r\n\\/]+)"
      "${shorthand_indicator_code}((${regex_cmake_identifier})(\\.${regex_cmake_identifier})*)"
      "${delimiter_code}=\"{\\1}\"${delimiter_code}"
      input
      "${input}")

  string(REGEX MATCHALL "(${code_fragment_regex})|(${literal_fragment_regex})"
               fragments "${input}")

  # decode escaped delimiters
  string(REPLACE "${delimiter_start_escape_code}" "${delimiter_start}"
                 fragments "${fragments}")
  string(REPLACE "${delimiter_end_escape_code}" "${delimiter_end}" fragments
                 "${fragments}")
  string(REPLACE "${shorthand_indicator_escape_code}" "${shorthand_indicator}"
                 fragments "${fragments}")
  string(REPLACE "${shorthand_indicator_code}" "${shorthand_hindicator}"
                 fragments "${fragments}")

  address_new()
  ans(result)
  address_append_string("${result}" "template_begin()")

  # set(result)
  foreach(fragment ${fragments})
    # message("${fragment}") decode brackets and semicolon in fragment now the
    # fragment input is exactly the same as it was in input
    string_decode_bracket("${fragment}")
    ans(fragment)

    string_decode_semicolon("${fragment}")
    ans(fragment)

    if("${fragment}" MATCHES "${code_fragment_regex}")
      # handle code fragment
      set(code "${CMAKE_MATCH_1}")

      if("${code}" MATCHES "^>>([a-zA-Z_0-9]+ )(.*)")
        set(func "${CMAKE_MATCH_1}")
        cmake_string_escape("${CMAKE_MATCH_2}")
        ans(output)
        set(code "${func}(${output})\n")
      elseif("${code}" MATCHES "^>([a-zA-Z_0-9]+ )(.*)")
        set(output_var "${CMAKE_MATCH_1}")
        cmake_string_escape("${CMAKE_MATCH_2}")
        ans(output)
        set(code "set(${output_var} \"${output}\")\n")
      endif()

      # special case <%=
      if("${code}" MATCHES "^=(.*)")
        set(code "${CMAKE_MATCH_1}")
        if("${code}" MATCHES "${regex_cmake_function}")
          set(exprcode "expr(\"${CMAKE_MATCH_1}(${CMAKE_MATCH_2})\")")
          # message("'set(__ans) \n ${exprcode} \n
          # template_out(\"\${__ans}\")'")
          set(code "set(__ans) \n ${exprcode} \n template_out(\"\${__ans}\")")
        else()
          set(code "template_out_format(${code})")
        endif()
      else()

      endif()

      address_append_string("${result}" "\n${code}")

    else()
      # handle literal fragment
      cmake_string_escape("${fragment}")
      ans(fragment)
      address_append_string("${result}" "\ntemplate_out(\"${fragment}\")")
    endif()
  endforeach()

  address_append_string("${result}" "\ntemplate_end()")

  address_get(${result})
  ans(res)
  # _message("template_compile\n~~~~~~~~~~~~~~\n${res}\n~~~~~~~~~~~~~~")
  return_ref(res)
endfunction()
