## http_put() -> 
##
## flags:
##   --response     						flag will return the http response object
##
##   --json											flag will deserialize the result data
##
##   --exit-code    						flag will return the http clients return code 
##															non-zero indicates an error
##   --show-progress 						flag causes a console message which indicates 
##															the progress of the operation
##
##   --raw           					 	flag witll cause input to be sent raw 
##															if flag is not specified the input is serialized
##															to json before sending
##
##   --file <file>							flag PUT the specified file instead of the input
##
##   --silent-fail						 	flag causes function to return nothing if it fails
##															(only usable if --response was not set)
##
##   --timeout <n>						 	value 
##
##   --inactivity-timeout <n>  	value 
## 
## events:
##   on_http_put(<uri> <content>)-> <uri?>:
##     event is called before put request is performed
##     user may cancel event and return a modified uri 
##     which is used to perform the request 
function(http_put uri)
	set(args ${ARGN})

	list_extract_flag(args --response)
	ans(return_response)

	list_extract_labelled_value(args --timeout)
	ans(timeout)
	
	list_extract_labelled_value(args --inactivity-timeout)
	ans(inactivity_timeout)

	list_extract_flag(args --show-progress)
	ans(show_progress)

	list_extract_flag(args --exit-code)
	ans(return_return_code)

	list_extract_flag(args --json)
	ans(return_json)

	list_extract_flag(args --raw)
	ans(put_raw)

	list_extract_labelled_value(args --file)
	ans(put_file)

	list_extract_flag(args --silent-fail)
	ans(silent_fail)

	path_temp()
	ans(temp_file)

	if(put_file)
		path_qualify(put_file)
		set(content_file "${put_file}")
		if(NOT EXISTS "${content_file}")
			error("http_put - file does not exists ${content_file}")
			message(FATAL_ERROR "http_put - file does not exists ${content_file}")
		endif()
	else()
		if(NOT put_raw)
			data("${args}")
			ans(content)
			json_write("${temp_file}" ${content})
			set(content_file "${temp_file}")
		else()
			fwrite("${temp_file}" "${args}")
			set(content_file "${temp_file}")
		endif()

	endif()

	## emit on_http_put event 
	## 
	event_emit(on_http_put ${uri} ${content})
	ans(modified_uri)
	if(modified_uri)
		set(uri "${modified_uri}")
	endif()

	## delegate cmake flags in correct format to file command
	if(show_progress)
		set(show_progress SHOW_PROGRESS)
	endif()
	if(NOT "${timeout}_" STREQUAL "_")
		set(timeout TIMEOUT "${timeout}")
	endif()
	if(NOT "${inactivity_timeout}_" STREQUAL "_")
		set(inactivity_timeout INACTIVITY_TIMEOUT "${inactivity_timeout}")
	endif()

	## upload a file (this actaully does a http put request)
	file(UPLOAD 
		"${content_file}" 
		"${uri}" 
		STATUS client_result 
		LOG http_log 
		${show_progress}
		${timeout}
		${inactivity_timeout}
	)
	## parse http client status
	list_extract(client_result client_status client_message)
	if(EXISTS "${temp_file}")
		rm("${temp_file}")
	endif()

	if(return_return_code)
		return_ref(client_status)
	endif()

	## parse response from log since it is not downloaded
	set(response_content)
	if("${http_log}" MATCHES  "Response:\n(.*)\n\nDebug:\n")
		set(response_content "${CMAKE_MATCH_1}")
	endif()
	
	if(NOT return_response AND client_status)
		error("http_put failed: ${client_message} - ${client_status}")
		if(NOT silent_fail)
			message(FATAL_ERROR "http_put failed: ${client_message} - ${client_status}")
		endif()
		return()
	endif()


	if(return_json)
		json_deserialize("${response_content}")
		ans(response_content)
	endif()

	if(NOT return_response)
		return_ref(response_content)
	endif()

	## parse rest of response
	http_last_response_parse("${http_log}")
	ans(response)

	map_set(${response} content "${response_content}")
	map_set(${response} client_status "${client_status}")
	map_set(${response} client_message "${client_message}")

	return_ref(response)
endfunction()