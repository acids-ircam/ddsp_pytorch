## `(<string_input:<string>> <string_search:<string>> <string_replace:<string>>)-><string>`
##
## Replaces the first occurence of "string_search" with "string_replace" in the input string "string_input".
##
## **Examples**
##  set(input "abc")
##  string_replace_first("${input}" "a" "z") # => "zbc"
##  set(input "aac")
##  string_replace_first("${input}" "aa" "z") # => "zc"
##
##
function(string_replace_first string_input string_search string_replace)
	string(FIND "${string_input}" "${string_search}" index)
	
	if(${index} LESS 0 OR "${string_search}_" STREQUAL "_")
		return_ref(string_input)
	endif()

	string(LENGTH "${string_search}" search_length)
	string(SUBSTRING "${string_input}" 0 "${index}" part1)
	math(EXPR index "${index} + ${search_length}")
	string(SUBSTRING "${string_input}" "${index}" -1 part2)
	set(res "${part1}${string_replace}${part2}")

	return_ref(res)
endfunction()