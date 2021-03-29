## Maps - Structured Data in CMake



Maps are very versatile and are missing dearly from CMake in my opinion. Maps are references as is standard in many languages. They are signified by having properties which are adressed by keys and can take any value.

Due to the "variable variable" system (ie names of variables are string which can be generated from other variables) it is very easy to implement the map system. Under the hood a value is mapped by calling `address_set(${map}.${key})`.  



### Function List


* [is_map](#is_map)
* [map_append](#map_append)
* [map_append_string](#map_append_string)
* [map_append_unique](#map_append_unique)
* [map_delete](#map_delete)
* [map_duplicate](#map_duplicate)
* [map_get](#map_get)
* [map_get_special](#map_get_special)
* [map_has](#map_has)
* [map_keys](#map_keys)
* [map_new](#map_new)
* [map_remove](#map_remove)
* [map_remove_item](#map_remove_item)
* [map_set](#map_set)
* [map_set_hidden](#map_set_hidden)
* [map_set_special](#map_set_special)
* [map_tryget](#map_tryget)
* [dfs](#dfs)
* [dfs_callback](#dfs_callback)
* [map_conditional_default](#map_conditional_default)
* [map_conditional_evaluate](#map_conditional_evaluate)
* [map_conditional_if](#map_conditional_if)
* [map_conditional_predicate_eval](#map_conditional_predicate_eval)
* [map_conditional_single](#map_conditional_single)
* [map_conditional_switch](#map_conditional_switch)
* [list_match](#list_match)
* [map_all_paths](#map_all_paths)
* [map_at](#map_at)
* [map_capture](#map_capture)
* [map_capture_new](#map_capture_new)
* [map_clear](#map_clear)
* [map_coerce](#map_coerce)
* [map_copy_shallow](#map_copy_shallow)
* [map_count](#map_count)
* [map_defaults](#map_defaults)
* [map_ensure](#map_ensure)
* [map_extract](#map_extract)
* [map_fill](#map_fill)
* [map_flatten](#map_flatten)
* [map_from_keyvaluelist](#map_from_keyvaluelist)
* [map_get_default](#map_get_default)
* [map_get_map](#map_get_map)
* [map_has_all](#map_has_all)
* [map_has_any](#map_has_any)
* [map_invert](#map_invert)
* [map_isempty](#map_isempty)
* [map_key_at](#map_key_at)
* [map_keys_append](#map_keys_append)
* [map_keys_clear](#map_keys_clear)
* [map_keys_remove](#map_keys_remove)
* [map_keys_set](#map_keys_set)
* [map_keys_sort](#map_keys_sort)
* [map_match](#map_match)
* [map_match_properties](#map_match_properties)
* [map_matches](#map_matches)
* [map_omit](#map_omit)
* [map_omit_regex](#map_omit_regex)
* [map_overwrite](#map_overwrite)
* [map_pairs](#map_pairs)
* [test](#test)
* [map_path_get](#map_path_get)
* [map_path_set](#map_path_set)
* [map_peek_back](#map_peek_back)
* [map_peek_front](#map_peek_front)
* [map_pick](#map_pick)
* [map_pick_regex](#map_pick_regex)
* [map_pop_back](#map_pop_back)
* [map_pop_front](#map_pop_front)
* [map_promote](#map_promote)
* [map_property_length](#map_property_length)
* [map_push_back](#map_push_back)
* [map_push_front](#map_push_front)
* [map_rename](#map_rename)
* [map_set_default](#map_set_default)
* [map_to_keyvaluelist](#map_to_keyvaluelist)
* [map_to_valuelist](#map_to_valuelist)
* [map_unpack](#map_unpack)
* [map_values](#map_values)
* [mm](#mm)
* [map_iterator](#map_iterator)
* [map_iterator_break](#map_iterator_break)
* [map_iterator_next](#map_iterator_next)
* [map_dfs_references_once](#map_dfs_references_once)
* [map_import_properties](#map_import_properties)
* [map_import_properties_all](#map_import_properties_all)
* [map_match_obj](#map_match_obj)
* [map_clone](#map_clone)
* [map_clone_deep](#map_clone_deep)
* [map_clone_shallow](#map_clone_shallow)
* [map_equal](#map_equal)
* [map_equal_obj](#map_equal_obj)
* [map_foreach](#map_foreach)
* [map_issubsetof](#map_issubsetof)
* [map_merge](#map_merge)
* [map_permutate](#map_permutate)
* [map_union](#map_union)

### Function Descriptions

## <a name="is_map"></a> `is_map`

 `(<any>...)-><bool>`

 returns true iff the specified value is a map
 note to self: cannot make this a macro because string will be evaluated




## <a name="map_append"></a> `map_append`





## <a name="map_append_string"></a> `map_append_string`





## <a name="map_append_unique"></a> `map_append_unique`

 map_append_unique 
 
 appends values to the <map>.<prop> and ensures 
 that <map>.<prop> stays unique 




## <a name="map_delete"></a> `map_delete`





## <a name="map_duplicate"></a> `map_duplicate`





## <a name="map_get"></a> `map_get`





## <a name="map_get_special"></a> `map_get_special`





## <a name="map_has"></a> `map_has`





## <a name="map_keys"></a> `map_keys`





## <a name="map_new"></a> `map_new`

 optimized version




## <a name="map_remove"></a> `map_remove`





## <a name="map_remove_item"></a> `map_remove_item`

 map_remove_item

 removes the specified items from <map>.<prop>
 returns the number of items removed




## <a name="map_set"></a> `map_set`





## <a name="map_set_hidden"></a> `map_set_hidden`





## <a name="map_set_special"></a> `map_set_special`





## <a name="map_tryget"></a> `map_tryget`





## <a name="dfs"></a> `dfs`





## <a name="dfs_callback"></a> `dfs_callback`





## <a name="map_conditional_default"></a> `map_conditional_default`





## <a name="map_conditional_evaluate"></a> `map_conditional_evaluate`





## <a name="map_conditional_if"></a> `map_conditional_if`





## <a name="map_conditional_predicate_eval"></a> `map_conditional_predicate_eval`





## <a name="map_conditional_single"></a> `map_conditional_single`





## <a name="map_conditional_switch"></a> `map_conditional_switch`





## <a name="list_match"></a> `list_match`





## <a name="map_all_paths"></a> `map_all_paths`





## <a name="map_at"></a> `map_at`

 returns the value at idx




## <a name="map_capture"></a> `map_capture`

 captures the listed variables in the map




## <a name="map_capture_new"></a> `map_capture_new`

 captures a new map from the given variables
 example
 set(a 1)
 set(b 2)
 set(c 3)
 map_capture_new(a b c)
 ans(res)
 json_print(${res})
 --> 
 {
   "a":1,
   "b":2,
   "c":3 
 }




## <a name="map_clear"></a> `map_clear`





## <a name="map_coerce"></a> `map_coerce`

 if `mapOrDefaultValue` is a map then just returns the map
 if not `mapOrDefaultValue` is assigned to a new map under the specified defaultKey




## <a name="map_copy_shallow"></a> `map_copy_shallow`





## <a name="map_count"></a> `map_count`

 `(<map>)-><uint>`

 returns the number of elements for the specified map




## <a name="map_defaults"></a> `map_defaults`





## <a name="map_ensure"></a> `map_ensure`





## <a name="map_extract"></a> `map_extract`





## <a name="map_fill"></a> `map_fill`

 files non existing or null values of lhs with values of rhs




## <a name="map_flatten"></a> `map_flatten`





## <a name="map_from_keyvaluelist"></a> `map_from_keyvaluelist`





## <a name="map_get_default"></a> `map_get_default`

 `(<map> <key> <any...>)-><any...>`

 returns the value stored in map.key or 
 sets the value at map.key to ARGN and returns 
 the value




## <a name="map_get_map"></a> `map_get_map`

 `(<map> <key>)-><map>`

 returns a map for the specified key
 creating it if it does not exist





## <a name="map_has_all"></a> `map_has_all`





## <a name="map_has_any"></a> `map_has_any`





## <a name="map_invert"></a> `map_invert`





## <a name="map_isempty"></a> `map_isempty`





## <a name="map_key_at"></a> `map_key_at`

 returns the key at the specified position




## <a name="map_keys_append"></a> `map_keys_append`





## <a name="map_keys_clear"></a> `map_keys_clear`





## <a name="map_keys_remove"></a> `map_keys_remove`





## <a name="map_keys_set"></a> `map_keys_set`





## <a name="map_keys_sort"></a> `map_keys_sort`





## <a name="map_match"></a> `map_match`

 checks if all fields specified in actual rhs are equal to the values in expected lhs
 recursively checks submaps




## <a name="map_match_properties"></a> `map_match_properties`





## <a name="map_matches"></a> `map_matches`





## <a name="map_omit"></a> `map_omit`





## <a name="map_omit_regex"></a> `map_omit_regex`





## <a name="map_overwrite"></a> `map_overwrite`

 overwrites all values of lhs with rhs




## <a name="map_pairs"></a> `map_pairs`





## <a name="test"></a> `test`





## <a name="map_path_get"></a> `map_path_get`





## <a name="map_path_set"></a> `map_path_set`





## <a name="map_peek_back"></a> `map_peek_back`





## <a name="map_peek_front"></a> `map_peek_front`





## <a name="map_pick"></a> `map_pick`





## <a name="map_pick_regex"></a> `map_pick_regex`





## <a name="map_pop_back"></a> `map_pop_back`





## <a name="map_pop_front"></a> `map_pop_front`





## <a name="map_promote"></a> `map_promote`





## <a name="map_property_length"></a> `map_property_length`

 returns the length of the specified property




## <a name="map_push_back"></a> `map_push_back`





## <a name="map_push_front"></a> `map_push_front`





## <a name="map_rename"></a> `map_rename`

 renames a key in the specified map




## <a name="map_set_default"></a> `map_set_default`

 `()-><bool>`

 sets the value of the specified prop if it does not exist
 ie if map_has returns false for the specified property
 returns true iff value was set




## <a name="map_to_keyvaluelist"></a> `map_to_keyvaluelist`





## <a name="map_to_valuelist"></a> `map_to_valuelist`





## <a name="map_unpack"></a> `map_unpack`

 unpacks the specified reference to a map
 let a map be stored in the var 'themap'
 let it have the key/values a/1 b/2 c/3
 map_unpack(themap) will create the variables
 ${themap.a} contains 1
 ${themap.b} contains 2
 ${themap.c} contains 3




## <a name="map_values"></a> `map_values`





## <a name="mm"></a> `mm`

 function which generates a map 
 out of the passed args 
 or just returns the arg if it is already valid




## <a name="map_iterator"></a> `map_iterator`

 initializes a new mapiterator




## <a name="map_iterator_break"></a> `map_iterator_break`





## <a name="map_iterator_next"></a> `map_iterator_next`

 this function moves the map iterator to the next position
 and returns true if it was possible
 e.g.
 map_iterator_next(myiterator) 
 ans(ok) ## is true if iterator had a next element
 variables ${myiterator.key} and ${myiterator.value} are available




## <a name="map_dfs_references_once"></a> `map_dfs_references_once`





## <a name="map_import_properties"></a> `map_import_properties`

 imports the specified properties into the current scope
 e.g map = {a:1,b:2,c:3}
 map_import_properties(${map} a c)
 -> ${a} == 1 ${b} == 2




## <a name="map_import_properties_all"></a> `map_import_properties_all`

 
 imports all properties of map into local scope




## <a name="map_match_obj"></a> `map_match_obj`

 returns true if actual has all properties (and recursive properties)
 that expected has




## <a name="map_clone"></a> `map_clone`





## <a name="map_clone_deep"></a> `map_clone_deep`





## <a name="map_clone_shallow"></a> `map_clone_shallow`





## <a name="map_equal"></a> `map_equal`





## <a name="map_equal_obj"></a> `map_equal_obj`

 compares two maps for value equality
 lhs and rhs may be objectish 




## <a name="map_foreach"></a> `map_foreach`





## <a name="map_issubsetof"></a> `map_issubsetof`





## <a name="map_merge"></a> `map_merge`





## <a name="map_permutate"></a> `map_permutate`

 
 permutates the specified input map 
 takes every key of the input map and treats the value as a list
 the result is n maps which contain one value per key




## <a name="map_union"></a> `map_union`








