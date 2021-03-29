## `(<json code>)->{}`
##
## deserializes the specified json code. In combination with json there are a few things
## that need mention:
## * semicolons.  If you use semicolons in json then they will be deserialized as
##   ASCII 31 (Unit Separator) which allows cmake to know the difference to the semicolons in a list
##   if you want semicolons to appear in cmake then use a json array. You can always use `string_decode_semicolon()`
##   to obtain the string as it was in json
##   eg. `[1,2,3] => 1;2;3`  `"1;2;3" => 1${semicolon_code}2${semicolon_code}3`
## 
function(json_deserialize json)
  json4("${json}")
  return_ans()
endfunction()