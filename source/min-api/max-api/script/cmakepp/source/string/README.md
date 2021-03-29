## String Functions


`CMake` has one utility for working with string: `string()`. Inside of the jewel there is hidden lots of functionality which just has to be exposed correctly to the user.  I also wanted to expose this functionality in the `cmakepp` way of doing things (ie `return values`)

So I have created somewhat alot of functions which does things that you might need and alot of what you will probably never need - but feel good about because its there :)


### Function List


* [ascii_char](#ascii_char)
* [ascii_code](#ascii_code)
* [ascii_generate_table](#ascii_generate_table)
* [cmake_string_to_json](#cmake_string_to_json)
* [delimiters](#delimiters)
* [argument_escape](#argument_escape)
* [format](#format)
* [regex_search](#regex_search)
* [string_append_line_indented](#string_append_line_indented)
* [string_char_at](#string_char_at)
* [string_char_at_set](#string_char_at_set)
* [string_codes](#string_codes)
* [string_combine](#string_combine)
* [string_concat](#string_concat)
* [string_contains](#string_contains)
* [string_decode_delimited](#string_decode_delimited)
* [string_ends_with](#string_ends_with)
* [string_eval](#string_eval)
* [string_find](#string_find)
* [string_indent](#string_indent)
* [string_isempty](#string_isempty)
* [string_isnumeric](#string_isnumeric)
* [string_length](#string_length)
* [string_lines](#string_lines)
* [string_match](#string_match)
* [string_normalize](#string_normalize)
* [string_normalize_index](#string_normalize_index)
* [string_overlap](#string_overlap)
* [string_pad](#string_pad)
* [string_random](#string_random)
* [string_regex_escape](#string_regex_escape)
* [string_remove_beginning](#string_remove_beginning)
* [string_remove_ending](#string_remove_ending)
* [string_repeat](#string_repeat)
* [string_replace](#string_replace)
* [string_replace_first](#string_replace_first)
* [string_shorten](#string_shorten)
* [string_slice](#string_slice)
* [string_split](#string_split)
* [string_split_at_first](#string_split_at_first)
* [string_split_at_last](#string_split_at_last)
* [string_split_parts](#string_split_parts)
* [string_starts_with](#string_starts_with)
* [string_substring](#string_substring)
* [string_take](#string_take)
* [string_take_address](#string_take_address)
* [string_take_any_delimited](#string_take_any_delimited)
* [string_take_delimited](#string_take_delimited)
* [string_take_regex](#string_take_regex)
* [string_take_whitespace](#string_take_whitespace)
* [string_to_target_name](#string_to_target_name)
* [string_to_title](#string_to_title)
* [string_tolower](#string_tolower)
* [string_toupper](#string_toupper)
* [string_trim](#string_trim)
* [string_trim_to_difference](#string_trim_to_difference)

### Function Descriptions

## <a name="ascii_char"></a> `ascii_char`





## <a name="ascii_code"></a> `ascii_code`





## <a name="ascii_generate_table"></a> `ascii_generate_table`

 generates the ascii table and stores it in the global ascii_table variable  




## <a name="cmake_string_to_json"></a> `cmake_string_to_json`





## <a name="delimiters"></a> `delimiters`

 **`delimiters()->[delimiter_begin, delimiter_end]`**

 parses delimiters and retruns a list of length 2 containing the specified delimiters. 
 The usefullness of this function becomes apparent when you use [string_take_delimited](#string_take_delimited)
 





## <a name="argument_escape"></a> `argument_escape`





## <a name="format"></a> `format`

 [**`format(<template string>)-><string>`**](format.cmake)

 this function utilizes [`assign(...)`](#assign) to evaluate expressions which are enclosed in handlebars: `{` `}`
 

 *Examples*
 ```cmake
 # create a object
 obj("{a:1,b:[2,3,4,5,6],c:{d:3}}")
 ans(data)
 ## use format to print navigated expressiosn:
 format("{data.a} + {data.c.d} = {data.b[2]}") => "1 + 3 = 4"
 format("some numbers: {data.b[2:$]}") =>  "some numbers: 4;5;6"
 ...
 ```
 *Note:* You may not use ASCII-29 since it is used interally in this function. If you don't know what this means - don't worry
 





## <a name="regex_search"></a> `regex_search`





## <a name="string_append_line_indented"></a> `string_append_line_indented`





## <a name="string_char_at"></a> `string_char_at`

 `(<input:<string>> <index:<int>>)-><string>`

 Returns the character at the specified position (index). 
 Indexing of strings starts at 0. Indices less than -1 are translated into "length - |index|"

 *Examples*
 set(input "example")
 string_char_at("${input}" 3) # => "m"
 string_char_at("${input}"-3) # => "l"






## <a name="string_char_at_set"></a> `string_char_at_set`

 `(<input:<string>> <index:<int>> <char:<string>>)-><string>`

 Sets the character at the specified position (index) to the input 'char'. 
 Indexing of strings starts at 0. Indices less than -1 are translated into "length - |index|"
 
 **Examples**
  set(input "example")
  string_char_at_set("${input}" 0 "E")  # => "Example"
  string_char_at_set("${input}" 2 "A")  # => "exAmple"
  string_char_at_set("${input}" -2 "E") # => "examplE"
 





## <a name="string_codes"></a> `string_codes`





## <a name="string_combine"></a> `string_combine`

 combines the varargs into a string joining them with separator
 e.g. string_combine(, a b c) => "a,b,c"




## <a name="string_concat"></a> `string_concat`





## <a name="string_contains"></a> `string_contains`

 `(<str:<string>> <search:<string>>)-><bool>`
  
 Returns true if the input string "str" contains "search"

 **Examples**
  set(input "endswith")
  string_contains("${input}" "with") # => true
  string_contains("${input}" "swi") # => true






## <a name="string_decode_delimited"></a> `string_decode_delimited`

 tries to parse a delimited string
 returns either the original or the parsed delimited string
 delimiters can be specified via varargs
 see also string_take_delimited




## <a name="string_ends_with"></a> `string_ends_with`

 `(<str:<string>> <search:<string>>)-><bool>`
  
 Returns true if the input string "str" ends with "search"

 **Examples**
  set(input "endswith")
  string_ends_with("${input}" "with") # => true
  string_ends_with("${input}" "width") # => false






## <a name="string_eval"></a> `string_eval`

 evaluates the string <str> in the current scope
 this is done by macro variable expansion
 evaluates both ${} and @ style variables




## <a name="string_find"></a> `string_find`

 `(<str:<string>> <substr:<string>>)-><int>`
  
 Returns the position where the "substr" was found 
 in the input "str", otherwise -1. 
 NOTE: The flag REVERSE causes the last position of "substr"
       to be returned

 **Examples**
  set(input "endswith")
  string_find("${input}" "with") # => 4
  string_find("${input}" "swi") # => 3






## <a name="string_indent"></a> `string_indent`





## <a name="string_isempty"></a> `string_isempty`

 `(<str:<string>>)-><bool>`
  
 Returns true if the input string "str" is empty 
 Note: cmake evals "false", "no" which 
       destroys tests for real emtpiness

 **Examples**
  set(input "")
  string_isempty("${input}") # => true
  set(input "false")
  string_isempty("${input}") # => false






## <a name="string_isnumeric"></a> `string_isnumeric`

 `(<str:<string>>)-><bool>`
  
 Returns true if the input string "str" is a positive integer 
 including "0"

 **Examples**
  set(input "1")
  string_isnumeric("${input}") # => true
  set(input "-1")
  string_isnumeric("${input}") # => false






## <a name="string_length"></a> `string_length`

 `(<str:<string>>)-><int>`
  
 Returns the length of the input string "str"

 **Examples**
  set(input "a")
  string_length("${input}") # => 1
  set(input "ab c")
  string_length("${input}") # => 4






## <a name="string_lines"></a> `string_lines`

 `(<input:<string>>)-><string...>`
  
 Splits the specified string "input" into lines
 Caveat: The string would have to be semicolon encoded
         to correctly display lines with semicolons 

 **Examples**
  set(input "a\nb")
  string_lines("${input}") # => "a;b"
  set(input "a b\nc")
  string_lines("${input}") # => "a b;c"






## <a name="string_match"></a> `string_match`

 `(<input:<string>>)-><bool>`
  
 Evaluates string "str" against regex "regex".
 Returns true if it matches.

 **Examples**
  set(input "a?")
  string_match("${input}" "[a-z]+\\?") # => true
  set(input "a bc .")
  string_match("${input}" "^b") # => false






## <a name="string_normalize"></a> `string_normalize`

 `(<input:<string>>)-><string>`
  
 Replaces all non-alphanumerical characters in the string "input" with an underscore 

 **Examples**
  set(input "a?")
  string_normalize("${input}") # => "a_"
  set(input "a bc .")
  string_normalize("${input}") # => "a bc _"






## <a name="string_normalize_index"></a> `string_normalize_index`

 `(<str:<string>> <index:<int>>)-><int>`
  
 Normalizes the index "index" of a corresponding input string "str".
 Negative indices are transformed into positive values: length - |index|
 Returns -1 if index is out of bounds (index > length of string or length - |index| + 1 < 0)

 **Examples**
  set(input "abcd")
  string_normalize_index("${input}" 3) # => 3
  string_normalize_index("${input}" -2) # => 3






## <a name="string_overlap"></a> `string_overlap`

 `(<lhs:<string>> <rhs:<string>>)-><string>`
  
 Returns the overlapping part of input strings "lhs" and "rhs".
 Starts at first char and continues until chars don't match.

 **Examples**
  set(input1 "abcd")
  set(input2 "abyx")
  string_overlap("${input1}" "${input2}") # => "ab"
  set(input2 "wxyz")
  string_overlap("${input1}" "${input2}") # => ""






## <a name="string_pad"></a> `string_pad`

 `(<str:<string>> <len:<int>> <argn:<string>>)-><string>`
  
 Pads the specified string to be as long as specified length "len".
  - If the string is longer then nothing is padded
  - If no delimiter is specified than " " (space) is used
  - If "--prepend" is specified for "argn" the padding is inserted at the beginning of "str"

 **Examples**
  set(input "word")
  string_pad("${input}" 6) # => "word  "
  string_pad("${input}" 4) # => "word"






## <a name="string_random"></a> `string_random`

 `()-><string>`
  
 Returns a randomly generated string.
 TODO: implement

 **Examples**
  string_random() # =>






## <a name="string_regex_escape"></a> `string_regex_escape`

 `(<str:<string>>)-><string>`
  
 Escapes chars used by regex strings in the input string "str".
 Escaped characters: "\ / ] [ * . - ^ $ ? ) ( |"

 **Examples**
  set(input "()")
  string_regex_escape("${input}") # => "\(\)"
  set(input "no_escape")
  string_regex_escape("${input}") # => "no_escape"






## <a name="string_remove_beginning"></a> `string_remove_beginning`

 `(<original:<string>> <beginning:<string>>)-><string>`

 Removes the beginning "n"-chars of the string "original".
 Number of chars "n" is calculated based on string "beginning".

 **Examples**
  set(input "abc")
  string_remove_ending("${input}" "a") # => "ab"
  string_remove_ending("${input}" "ab") # => "a"






## <a name="string_remove_ending"></a> `string_remove_ending`

 `(<original:<string>> <ending:<string>>)-><string>`

 Removes the back "n"-chars of the string "original".
 Number of chars "n" is calculated based on string "ending".

 **Examples**
  set(input "abc")
  string_remove_ending("${input}" "a") # => "ab"
  string_remove_ending("${input}" "ab") # => "a"






## <a name="string_repeat"></a> `string_repeat`

 `(<what:<string>> <n:<int>>)-><string>`

 Repeats string "what" "n" times and separates them with an optional separator

 **Examples**
  set(input "a")
  string_repeat("${input}" 2) # => "aa"
  string_repeat("${input}" 2 "@") # => "a@a"

  




## <a name="string_replace"></a> `string_replace`

 `(<str:<string>> <pattern:<string>> <replace:<string>>)-><string>`

 Replaces all occurences of "pattern" with "replace" in the input string "str".

 **Examples**
  set(input "abca")
  string_replace("a" "z" "${input}") # => "zbcz"
  set(input "aaa")
  string_replace("a" "z" "${input}") # => "zzz"






## <a name="string_replace_first"></a> `string_replace_first`

 `(<string_input:<string>> <string_search:<string>> <string_replace:<string>>)-><string>`

 Replaces the first occurence of "string_search" with "string_replace" in the input string "string_input".

 **Examples**
  set(input "abc")
  string_replace_first("${input}" "a" "z") # => "zbc"
  set(input "aac")
  string_replace_first("${input}" "aa" "z") # => "zc"






## <a name="string_shorten"></a> `string_shorten`

 `(<str:<string>> <max_length:<int>>)-><string>`

 Shortens the string "str" to be at most "max_length" characters long.
 Note on "max_length": max_length includes the shortener string (default 3 chars "...").
 Returns the result in "res".

 **Examples**
  set(input "abcde")
  string_shorten("${input}" 4) # => "a..."
  string_shorten("${input}" 3) # => "..."
  string_shorten("${input}" 2) # => ""
  string_shorten("${input}" 2 ".") # => "a."






## <a name="string_slice"></a> `string_slice`

 `(<str:<string>> <start_index:<int>> <end_index:<int>>)-><string>`

 Extracts a portion from string "str" at the specified index: [start_index, end_index)
 Indexing of slices starts at 0. Indices less than -1 are translated into "length - |index|"
 Returns the result in "result".

 **Examples**
  set(input "abc")
  string_slice("${input}" 0 1) # => "a"
  set(input "abc")
  string_slice("${input}" 0 2) # => "ab"






## <a name="string_split"></a> `string_split`

 `(<string_subject:<string>> <split_regex:<string>>)-><string...>`

 Splits the string "input" at the occurence of the regex "split_regex".
 Returns the result in "res".
 TODO: does not handle strings containing list separators properly

 **Examples**
  set(input "a@b@c")
  string_split("${input}" "@") # => "a;b;c"






## <a name="string_split_at_first"></a> `string_split_at_first`

 `(<parta:<string&>> <partb:<string&>> <input:<string>> <separator:<string>>)-><parta:<string&>> <partb:<string&>>`

 Splits the string "input" at the first occurence of "separator" and returns 
 both parts in the string references "parta" and "partb".
 See **Examples** for passing references.

 **Examples**
 
  set(input "a@b@c")
  string_split_at_first(partA partB "${input}" "@") # => partA equals "a", partB equals "b@c"






## <a name="string_split_at_last"></a> `string_split_at_last`

 `(<parta:<string&>> <partb:<string&>> <input:<string>> <separator:<string>>)-><parta:<string&>> <partb:<string&>>`

 Splits the string "input" at the last occurence of "separator" and returns 
 both parts in the string references "parta" and "partb".
 See **Examples** for passing references.

 **Examples**
  set(input "a@b@c")
  string_split_at_last(partA partB "${input}" "@") # => partA equals "a@b", partB equals "c"






## <a name="string_split_parts"></a> `string_split_parts`

 `(<str:<string>> <length:<int>>)-><first_node:<linked list>>`

 Splits the string "str" into multiple parts of length "length". 
 Returns a linked list of the parts

 **Examples**
  set(input "abc")
  string_split_parts("${input}" 1) # => linked_list("a", "b", "c")
  string_split_parts("${input}" 2) # => linked_list("ab", "c")
  string_split_parts("${input}" 3) # => linked_list("abc")






## <a name="string_starts_with"></a> `string_starts_with`

 `(<str:<string>> <search:<string>>)-><bool>`

 Returns true if "str" starts with the string "search"
 
 **Examples**
  string_starts_with("substring" "sub") # => true
  string_starts_with("substring" "ub") # => false






## <a name="string_substring"></a> `string_substring`

 `(<str:<string>> <start:<int>> <end:<int>>)-><string>`

 Wrapper function for substring.
 Returns a substring of input "str" with the index parameter "start" and optionally "len".
 Note on indexing: len is the amount of chars to be extracted starting from index "start"
 
 **Examples**
  string_substring("substring" 1)     # => "ubstring"
  string_substring("substring" 1 2)   # => "ub"
  string_substring("substring" -3 2)  # => "ng"






## <a name="string_take"></a> `string_take`

 `(<str_name:<string&>> <match:<string>>)-><str_name:<string&>> string>`

 Removes "match" from a string reference "str_name" and returns the "match" string.
 Only matches from the beginning of the string reference.
 
 **Examples**
  set(input "word")
  string_take(input "w") # => input equals "ord", match equals "w"
  set(input "word")
  string_take(input "ord") # => input is unchanged, no match is returned






## <a name="string_take_address"></a> `string_take_address`

 `(<str_ref:<string&>>)-><str_ref:<string&>> <string>`

 Removes an address (regex format: ":[1-9][0-9]*") from a string reference and returns the address in "res".
 The address is also removed from the input string reference (str_ref).

 **Examples**






## <a name="string_take_any_delimited"></a> `string_take_any_delimited`

 `(<str_ref:<string&>> <delimiters:<delimiter:<string>>...>>)-><str_ref:<string&>> <string>`

 Removes delimiters of a string and the undelimited string is returned.
 The undelimited string is also removed from the input string reference (__str_ref).
 Notes on the delimiter:
  - Can be a list of delimiters
  - Beginning and end delimiter can be specified
  - May only be a single char
  - Escaped delimiters are unescaped

 **Examples**
  set(in_ref_str "'a string'")
  string_take_any_delimited(in_ref_str ') # => in_ref_str equals "" and match equals "a string"
  set(in_ref_str "\"a string\", <another one>")
  string_take_any_delimited(in_ref_str "'', <>") # => in_ref_str equals "\"a string\"" and match equals "another one"






## <a name="string_take_delimited"></a> `string_take_delimited`

 `(<__str_ref:<string&>>)-><__str_ref:<string&>> <string>`

 Removes delimiters of a string and the undelimited string is returned.
 The undelimited string is also removed from the input string reference (__str_ref).
 Notes on the delimiter:
  - Default is double quote ""
  - Beginning and end delimiter can be specified
  - May only be a single char
  - Escaped delimiters are unescaped

 **Examples**
  set(in_ref_str "'a string'")
  string_take_delimited(in_ref_str ') # => in_ref_str equals "" and res equals "a string"
  set(in_ref_str "'a string'")
  string_take_delimited(in_ref_str "''") # => same as above






## <a name="string_take_regex"></a> `string_take_regex`

 `(<str_name:<string&>> <regex:<string>> <replace:<string>>)-><str_name:<string&>> <string>`

 Tries to match the regex at the begging of ${${str_name}} and returns the match.
 Side effect: Input reference ${str_name} is shortened in the process.
 See **Examples** for passing references.

 **Examples**
  set(in_ref_str "keep_two_whitespaces  ")
  string_take_regex(in_ref_str "[^ ]*" "") # => in_ref_str equals "  "






## <a name="string_take_whitespace"></a> `string_take_whitespace`

 `(<__str_ref:<string&>>)-><__str_ref:<string&>>`

 Removes preceeding whitespaces of the input string reference.
 See **Examples** for passing references.

 **Examples**
  set(in_ref_str "   test")
  string_take_whitespace(in_ref_str) # => in_ref_str equals "test"
  set(in_ref_str "   test  ")
  string_take_whitespace(in_ref_str) # => in_ref_str equals "test  "






## <a name="string_to_target_name"></a> `string_to_target_name`





## <a name="string_to_title"></a> `string_to_title`

 `(<input:<string>>)-><string>`

 Transforms the input string to title case.
 Tries to be smart and keeps some words small.
 List of words that are kept small:
 "a, an, and, as, at, but, by, en, for, if, in, of, on, or, the, to, via, vs, v, v., vs."

 **Examples**
  set(input "the function string_totitle works")
  string_to_title("${input}") # => "The Function string_totitle Works"
  set(input "testing a small word")
  string_to_title("${input}") # => "Testing a Small Word"






## <a name="string_tolower"></a> `string_tolower`

 `(<input:<string>>)-><string>`

 Transforms the specified string to lower case.
 
 **Examples**
  string_tolower("UPPER") # => "upper"






## <a name="string_toupper"></a> `string_toupper`

 `(<input:<string>>)-><string>`

 Transforms the specified string to upper case.
 
 **Examples**
  string_tolower("lower") # => "LOWER"






## <a name="string_trim"></a> `string_trim`

 `(<input:<string>>)-><string>`

 Trims the string, by removing whitespaces at the beginning and end.
 
 **Examples**
  string_tolower("  whitespaces  ") # => "whitespaces"






## <a name="string_trim_to_difference"></a> `string_trim_to_difference`

 `(<lhs:<string&>> <rhs:<string&>>)-><lhs:<string&>> <rhs:<string&>>`

 Removes the beginning of the string that matches
 from reference string "lhs" and "rhs". 
 See **Examples** for passing references.

 **Examples**
  set(in_lhs "simple test")
  set(in_rhs "simple a")
  string_trim_to_difference(in_lhs in_rhs) # => in_lhs equals "test", in_rhs equals "a" 
  set(in_lhs "a test")
  set(in_rhs "b test")
  string_trim_to_difference(in_lhs in_rhs) # => in_lhs equals "a test", in_rhs equals "b test" 








