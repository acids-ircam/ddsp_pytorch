function(json_tokenize result json)

	set(regex "(\\{|\\}|:|,|\\[|\\]|\"(\\\\.|[^\"])*\")")
	string(REGEX MATCHALL "${regex}" matches "${json}")


	# replace brackets with angular brackets because
	# normal brackes are not handled properly by cmake
	string(REPLACE  ";[;" ";<;" matches "${matches}")
	string(REPLACE ";];" ";>;" matches "${matches}")
	string(REPLACE "[" "†" matches "${matches}")
	string(REPLACE "]" "‡" matches "${matches}")

	set(tokens)
	foreach(match ${matches})
		string_char_at("${match}" 0)
		ans(char)
		if("${char}" STREQUAL "[")
			string_char_at("${match}" -2)
			ans(char)
			if(NOT "${char}" STREQUAL "]")
				message(FATAL_ERROR "json syntax error: no closing ']' instead: '${char}' ")
			endif()
			string(LENGTH "${match}" len)
			math(EXPR len "${len} - 2")
			string(SUBSTRING ${match} 1 ${len} array_values)
			set(tokens ${tokens} "<")
			foreach(submatch ${array_values})
				set(tokens ${tokens} ${submatch} )
			endforeach()
			set(tokens ${tokens} ">")
		else()
			set(tokens ${tokens} ${match})
		endif()
	endforeach()

	set(${result} ${tokens} PARENT_SCOPE)
endfunction()