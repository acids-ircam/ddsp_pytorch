# Uniform Resource Identifiers (URIs)

Uniform Resource Identifiers are often used for more than just internet addresses.  They are able to identify any type of resource and are truly cross platform.  Even something as simple as parsing a path can take on complex forms in edge cases.  My motivation for writing an URI parser was that I needed a sure way to identify a path in a command line call. 

My work is based arround [RFC2396](https://www.ietf.org/rfc/rfc2396.txt) by Berners Lee et al.  The standard is enhanced by allowing delimited URIs and Windows Paths as URIs. You can always turn this behaviour off however and use flags to use the pure standard.

URI parsing with cmake is not something you should do thousands of times because alot of regex calls go into generating an uri object.

## Example

*Parse an URI and print it to the Console*
```
uri("https://www.google.de/u/0/mail/?arg1=123&arg2=arg4&arg3.arg5=2#readmails some other data")
ans(res)
json_print(${res})
```

*output*
```
{
 "input":"https://www.google.de/u/0/mail/?arg1=123&arg2=arg4&arg3.arg5=2#readmails some other data",
 "uri":"https://www.google.de/u/0/mail/?arg1=123&arg2=arg4&arg3.arg5=2#readmails",
 "rest":" some other data",
 "delimited_rest":null,
 "delimiters":null,
 "windows_absolute_path":false,
 "scheme":"https",
 "scheme_specific_part":"//www.google.de/u/0/mail/?arg1=123&arg2=arg4&arg3.arg5=2#readmails",
 "net_path":"www.google.de/u/0/mail/",
 "authority":"www.google.de",
 "path":"/u/0/mail/",
 "query":"arg1=123&arg2=arg4&arg3.arg5=2",
 "fragment":"readmails",
 "schemes":"https",
 "user_info":null,
 "user_name":null,
 "password":null,
 "host_port":"www.google.de",
 "host":"www.google.de",
 "normalized_host":"www.google.de",
 "labels":[
  "www",
  "google",
  "de"
 ],
 "top_label":"de",
 "domain":"google.de",
 "ip":null,
 "port":null,
 "segments":[
  "u",
  0,
  "mail"
 ],
 "encoded_segments":[
  "u",
  0,
  "mail"
 ],
 "last_segment":"mail",
 "trailing_slash":"/",
 "leading_slash":"/",
 "normalized_segments":[
  "u",
  0,
  "mail"
 ],
 "file":"mail",
 "extension":null,
 "file_name":"mail",
 "params":{
  "arg1":123,
  "arg2":"arg4",
  "arg3":{
   "arg5":2
  }
 }
}
```

*Absolute Windows Path*

```
# output for `C:\windows\path.txt`
{
 "input":"c:\\windows\\path.txt",
 "uri":"file:///c:/windows/path.txt",
 "rest":null,
 "delimited_rest":null,
 "delimiters":null,
 "windows_absolute_path":true,
 "scheme":"file",
 "scheme_specific_part":"///c:/windows/path.txt",
 "net_path":"/c:/windows/path.txt",
 "authority":null,
 "path":"/c:/windows/path.txt",
 "query":null,
 "fragment":null,
 "schemes":"file",
 "user_info":null,
 "user_name":null,
 "password":null,
 "host_port":null,
 "host":null,
 "normalized_host":"localhost",
 "labels":null,
 "top_label":null,
 "domain":null,
 "ip":null,
 "port":null,
 "segments":[
  "c:",
  "windows",
  "path.txt"
 ],
 "encoded_segments":[
  "c:",
  "windows",
  "path.txt"
 ],
 "last_segment":"path.txt",
 "trailing_slash":null,
 "leading_slash":"/",
 "normalized_segments":[
  "c:",
  "windows",
  "path.txt"
 ],
 "file":"path.txt",
 "extension":"txt",
 "file_name":"path",
 "params":{
 }
}
```


*Perverted Example*
```
uri("'scheme1+http://user:password@@102.13.44.22:23%54/C:\\Program Files(x86)/dir number 1\\file.text.txt?asd=23#asd'")
ans(res)
json_print(${res})
```
*output*
```
{
 "input":"'scheme1+http://user:password@@102.13.44.22:23%54/C:\\Program Files(x86)/dir number 1\\file.text.txt?asd=23#asd'",
 "uri":"scheme1+http://user:password@@102.13.44.22:23%54/C:/Program%20Files(x86)/dir%20number%201/file.text.txt?asd=23#asd",
 "rest":null,
 "delimited_rest":null,
 "delimiters":null,
 "windows_absolute_path":false,
 "scheme":"scheme1+http",
 "scheme_specific_part":"//user:password@@102.13.44.22:23%54/C:/Program%20Files(x86)/dir%20number%201/file.text.txt?asd=23#asd",
 "net_path":"user:password@@102.13.44.22:23%54/C:/Program%20Files(x86)/dir%20number%201/file.text.txt",
 "authority":"user:password@@102.13.44.22:23%54",
 "path":"/C:/Program%20Files(x86)/dir%20number%201/file.text.txt",
 "query":"asd=23",
 "fragment":"asd",
 "schemes":[
  "scheme1",
  "http"
 ],
 "user_info":"user:password",
 "user_name":"user",
 "password":"password",
 "host_port":"@102.13.44.22:23%54",
 "host":"@102.13.44.22",
 "normalized_host":"@102.13.44.22",
 "labels":[
  "@102",
  13,
  44,
  22
 ],
 "top_label":22,
 "domain":44.22,
 "ip":null,
 "port":23,
 "segments":[
  "C:",
  "Program Files(x86)",
  "dir number 1",
  "file.text.txt"
 ],
 "encoded_segments":[
  "C:",
  "Program%20Files(x86)",
  "dir%20number%201",
  "file.text.txt"
 ],
 "last_segment":"file.text.txt",
 "trailing_slash":null,
 "leading_slash":"/",
 "normalized_segments":[
  "C:",
  "Program Files(x86)",
  "dir number 1",
  "file.text.txt"
 ],
 "file":"file.text.txt",
 "extension":"txt",
 "file_name":"file.text",
 "params":{
  "asd":23
 }
}
```

## Caveats

* Parsing is always a performance problem in CMake therfore parsing URIs is also a performance problem don't got parsing thousands of uris. I Tried to parse 100 URIs on my MBP 2011 and it took 6716 ms so 67ms to parse a single uri
* Non standard behaviour can always ensue when working with file paths and spaces and windows.  However this is the closest I came to having a general solution




## Function List


* [dns_parse](#dns_parse)
* [uri](#uri)
* [uri_check_scheme](#uri_check_scheme)
* [uri_coerce](#uri_coerce)
* [uri_decode](#uri_decode)
* [uri_encode](#uri_encode)
* [uri_format](#uri_format)
* [uri_normalize_input](#uri_normalize_input)
* [uri_params_deserialize](#uri_params_deserialize)
* [uri_params_serialize](#uri_params_serialize)
* [uri_parse](#uri_parse)
* [uri_parse_authority](#uri_parse_authority)
* [uri_parse_file](#uri_parse_file)
* [uri_parse_path](#uri_parse_path)
* [uri_parse_query](#uri_parse_query)
* [uri_parse_scheme](#uri_parse_scheme)
* [uri_qualify_local_path](#uri_qualify_local_path)
* [uri_recommended_to_escape](#uri_recommended_to_escape)
* [uri_remove_schemes](#uri_remove_schemes)
* [uri_to_localpath](#uri_to_localpath)

## Function Descriptions

## <a name="dns_parse"></a> `dns_parse`





## <a name="uri"></a> `uri`





## <a name="uri_check_scheme"></a> `uri_check_scheme`

 
 checks to see if all specified items are in list 
 using list_check_items
 




## <a name="uri_coerce"></a> `uri_coerce`


 forces the specified variable reference to become an uri




## <a name="uri_decode"></a> `uri_decode`

 decodes an uri encoded string ie replacing codes %XX with their ascii values




## <a name="uri_encode"></a> `uri_encode`

 encodes a string to uri format 
 if you can pass decimal character codes  which are encoded 
 if you do not pass any codes  the characters  recommended by rfc2396
 are encoded




## <a name="uri_format"></a> `uri_format`





## <a name="uri_normalize_input"></a> `uri_normalize_input`

 normalizes the input for the uri
 expects <uri> to have a property called input
 ensures a property called uri is added to <uri> which contains a valid uri string 




## <a name="uri_params_deserialize"></a> `uri_params_deserialize`





## <a name="uri_params_serialize"></a> `uri_params_serialize`





## <a name="uri_parse"></a> `uri_parse`

 parses an uri
 input can be any path or uri
 whitespaces in segments are allowed if string is delimited by double or single quotes(non standard behaviour)
{




## <a name="uri_parse_authority"></a> `uri_parse_authority`





## <a name="uri_parse_file"></a> `uri_parse_file`

 expects last_segment property to exist
 ensures file_name, file, extension exists




## <a name="uri_parse_path"></a> `uri_parse_path`





## <a name="uri_parse_query"></a> `uri_parse_query`

 parses the query field of uri and sets  the uri.params field to the parsed data




## <a name="uri_parse_scheme"></a> `uri_parse_scheme`





## <a name="uri_qualify_local_path"></a> `uri_qualify_local_path`

 tries to interpret the uri as a local path and replaces it 
 with a normalized local path (ie file:// ...)
 returns a new uri




## <a name="uri_recommended_to_escape"></a> `uri_recommended_to_escape`

 characters specified in rfc2396
 37 %  (percent)
 126 ~ (tilde) 
 1-32 (control chars) (nul is not allowed) 
 127 (del)
 32 (space)
 35 (#) sharp fragment identifer
 60 (<) 62 (>) 34 (") delimiters 
 unwise 
 123 { 125 } 124 | 92 \ 94 ^ 91 [ 93 ] 96 `




## <a name="uri_remove_schemes"></a> `uri_remove_schemes`





## <a name="uri_to_localpath"></a> `uri_to_localpath`

 formats an <uri~> to a localpath 






