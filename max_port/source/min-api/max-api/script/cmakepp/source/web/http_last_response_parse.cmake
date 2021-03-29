
## returns a response object for the last response in the specified http_log
## http_log is returned by cmake's file(DOWNLOAD|PUT LOG) function
## layout
## {
##   http_version:
##   http_status_code:
##   http_reason_phrase:
##   http_headers:{}
##   http_request:{
##      http_version:	
##      http_request_url:	
##      http_method:
##      http_headers:{}	
##   }
## }
function(http_last_response_parse http_log)
	string_encode_semicolon("${http_log}")
	ans(http_log)
	http_regexes()
	
	string(REGEX MATCHALL "(${http_request_header_regex})" requests "${http_log}")
	string(REGEX MATCHALL "(${http_response_header_regex})" responses "${http_log}")

	list_pop_back(requests)
	ans(request)
	http_request_header_parse("${request}")
	ans(request)

	list_pop_back(responses)
	ans(response)

	http_response_header_parse("${response}")
	ans(response)
	map_set(${response} http_request "${request}")
	return_ref(response)
endfunction()