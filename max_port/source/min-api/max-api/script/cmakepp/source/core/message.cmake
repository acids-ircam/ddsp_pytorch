
function(message)
	cmake_parse_arguments("" "PUSH_AFTER;POP_AFTER;DEBUG;INFO;FORMAT;PUSH;POP" "LEVEL" "" ${ARGN})
	set(log_level ${_LEVEL})
	set(text ${_UNPARSED_ARGUMENTS})

	## indentation
	if(_PUSH)
		message_indent_push()
	endif()
	if(_POP)
		message_indent_pop()
	endif()


	message_indent_get()
	ans(indent)
	if(_POP_AFTER)
		message_indent_pop()
	endif()
	if(_PUSH_AFTER)
		message_indent_push()
	endif()
	## end of indentationb


	## log_level
	if(_DEBUG)
		if(NOT log_level)
			set(log_level 3)
		endif()
		set(text STATUS ${text})
	endif()
	if(_INFO)
		if(NOT log_level)
			set(log_level 2)
		endif()
		set(text STATUS ${text})
	endif()
	if(NOT log_level)
		set(log_level 0)
	endif()

	if(NOT MESSAGE_LEVEL)
		set(MESSAGE_LEVEL 3)
	endif()

	list(GET text 0 modifier)
	if(${modifier} MATCHES "FATAL_ERROR|STATUS|AUTHOR_WARNING|WARNING|SEND_ERROR|DEPRECATION")
		list(REMOVE_AT text 0)
	else()
		set(modifier)
	endif()

	## format
	if(_FORMAT)
		format( "${text}")
		ans(text)
	endif()

	if(NOT MESSAGE_DEPTH )
		set(MESSAGE_DEPTH -1)
	endif()

	if(NOT text)
		return()
	endif()

	map_new()
	ans(message)
	map_set(${message} text "${text}")
	map_set(${message} indent_level ${message_indent_level})
	map_set(${message} log_level ${log_level})
	map_set(${message} mode "${modifier}")
	event_emit(on_message ${message})

	if(log_level GREATER MESSAGE_LEVEL)
		return()
	endif()
	if(MESSAGE_QUIET)
		return()
	endif()
	# check if deep message are to be ignored
	if(NOT MESSAGE_DEPTH LESS 0)
		if("${message_indent_level}" GREATER "${MESSAGE_DEPTH}")
			return()
		endif()
	endif()

	tock()

	## clear status line
	status_line_clear()
	_message(${modifier} "${indent}" "${text}")
	status_line_restore()

	
	return()
endfunction()


