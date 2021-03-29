## Json Serialization and Deserialization


## <a name="json"></a>Json Serialziation and Deserialization

I have written five functions which you can use to serialize and deserialize json.  

### Functions

* `json(<map>)` 
    - transforms the specified object graph to condensed json (no superfluos whitespace)
    - cycles are detected but not handled. (the property will not be set if it would cause a cycle e.g. map with a self reference would be serialized to `{"selfref":}` which is incorrect json... which will be addressed in the future )  
    - unicode is not transformed into `\uxxxx` because of lacking unicode support in cmake
* `json_indented(<map>)` 
    - same as `json()` however is formatted to be readable ie instead of `{"var":"val","obj":{"var":["arr","ay"]}}` it will be 
```
{
  "var":"var",
  "obj":[
    "arr",
    "ay"
  ]
}
```
* `json_deserialize(<json_string>)`
  - deserialized a json string ignoring any unicode escapes (`\uxxxx`)
* `json_read(<file>)`
  - directly deserializes af json file into a map
* `json_write(<file> <map>)`
  - write the map to the file

### Caveats
As can be seen in the functions' descriptions unicode is not support. Also you should probably avoid cycles.  

### Caching
Because deserialization is extremely slow I chose to cache the results of deserialization. So the first time you deserialize something large it might take long however the next time it will be fast (if it hasn't changed).
This is done by creating a hash from the input string and using it as a cache key. The cache is file based using Quick Map Syntax (which is alot faster to parse since it only has to be included by cmake).  



### Function List


* [json](#json)
* [json2](#json2)
* [json2_definition](#json2_definition)
* [json3_cached](#json3_cached)
* [json4](#json4)
* [json_deserialize](#json_deserialize)
* [json_escape](#json_escape)
* [json_extract_string_value](#json_extract_string_value)
* [json_format_tokens](#json_format_tokens)
* [json_indented](#json_indented)
* [json_print](#json_print)
* [json_read](#json_read)
* [json_string_to_cmake](#json_string_to_cmake)
* [json_tokenize](#json_tokenize)
* [json_write](#json_write)

### Function Descriptions

## <a name="json"></a> `json`





## <a name="json2"></a> `json2`





## <a name="json2_definition"></a> `json2_definition`





## <a name="json3_cached"></a> `json3_cached`





## <a name="json4"></a> `json4`





## <a name="json_deserialize"></a> `json_deserialize`

 `(<json code>)->{}`

 deserializes the specified json code. In combination with json there are a few things
 that need mention:
 * semicolons.  If you use semicolons in json then they will be deserialized as
   ASCII 31 (Unit Separator) which allows cmake to know the difference to the semicolons in a list
   if you want semicolons to appear in cmake then use a json array. You can always use `string_decode_semicolon()`
   to obtain the string as it was in json
   eg. `[1,2,3] => 1;2;3`  `"1;2;3" => 1${semicolon_code}2${semicolon_code}3`
 




## <a name="json_escape"></a> `json_escape`





## <a name="json_extract_string_value"></a> `json_extract_string_value`

 quickly extracts string properties values from a json string
 useful for large json files with unique property keys




## <a name="json_format_tokens"></a> `json_format_tokens`





## <a name="json_indented"></a> `json_indented`





## <a name="json_print"></a> `json_print`





## <a name="json_read"></a> `json_read`





## <a name="json_string_to_cmake"></a> `json_string_to_cmake`





## <a name="json_tokenize"></a> `json_tokenize`





## <a name="json_write"></a> `json_write`







