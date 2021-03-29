## [**`format(<template string>)-><string>`**](<%="${template_path}"%>)
##
## this function utilizes [`assign(...)`](#assign) to evaluate expressions which are enclosed in handlebars: `{` `}`
## 
##
## *Examples*
## ```cmake
## # create a object
## obj("{a:1,b:[2,3,4,5,6],c:{d:3}}")
## ans(data)
## ## use format to print navigated expressiosn:
## format("{data.a} + {data.c.d} = {data.b[2]}") => "1 + 3 = 4"
## format("some numbers: {data.b[2:$]}") =>  "some numbers: 4;5;6"
## ...
## ```
## *Note:* You may not use ASCII-29 since it is used interally in this function. If you don't know what this means - don't worry
## 
##
function(format)
  string(ASCII 29 delimiter)
  set(template "${ARGN}")
  string(REGEX MATCHALL "{[^}]*}" matches "${template}")
  list_remove_duplicates(matches)
  foreach(match ${matches})
    string(REGEX REPLACE "^{(.*)}$" "\\1" match "${match}")
    assign(value = ${match})
    string(REPLACE "{${match}}" "${value}" template "${template}")
  endforeach()
  return_ref(template)
endfunction()
