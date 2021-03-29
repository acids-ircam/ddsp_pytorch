# Collections

`CMake` is missing a lot of helper functions when it comes to collections. 
However using the `list` function that `CMake` provides it is possible to add alot of functions that help the developer.




### Function List


* [encoded_list](#encoded_list)
* [encoded_list_append](#encoded_list_append)
* [encoded_list_decode](#encoded_list_decode)
* [encoded_list_get](#encoded_list_get)
* [encoded_list_peek_back](#encoded_list_peek_back)
* [encoded_list_peek_front](#encoded_list_peek_front)
* [encoded_list_pop_back](#encoded_list_pop_back)
* [encoded_list_pop_front](#encoded_list_pop_front)
* [encoded_list_remove_at](#encoded_list_remove_at)
* [encoded_list_remove_item](#encoded_list_remove_item)
* [encoded_list_set](#encoded_list_set)
* [encoded_list_to_cmake_string](#encoded_list_to_cmake_string)
* [is_encoded_list](#is_encoded_list)
* [index_range](#index_range)
* [linked_list_insert_after](#linked_list_insert_after)
* [linked_list_insert_before](#linked_list_insert_before)
* [linked_list_new](#linked_list_new)
* [linked_list_node_new](#linked_list_node_new)
* [linked_list_peek_back](#linked_list_peek_back)
* [linked_list_peek_front](#linked_list_peek_front)
* [linked_list_pop_back](#linked_list_pop_back)
* [linked_list_pop_front](#linked_list_pop_front)
* [linked_list_push_back](#linked_list_push_back)
* [linked_list_push_front](#linked_list_push_front)
* [linked_list_remove](#linked_list_remove)
* [linked_list_replace](#linked_list_replace)
* [list_after](#list_after)
* [list_all](#list_all)
* [list_any](#list_any)
* [list_append](#list_append)
* [list_at](#list_at)
* [list_before](#list_before)
* [list_check_items](#list_check_items)
* [list_combinations](#list_combinations)
* [list_contains](#list_contains)
* [list_contains_any](#list_contains_any)
* [list_count](#list_count)
* [list_equal](#list_equal)
* [list_erase](#list_erase)
* [list_erase_slice](#list_erase_slice)
* [list_except](#list_except)
* [list_extract](#list_extract)
* [list_extract_any_flag](#list_extract_any_flag)
* [list_extract_any_labelled_value](#list_extract_any_labelled_value)
* [list_extract_flag](#list_extract_flag)
* [list_extract_flag_name](#list_extract_flag_name)
* [list_extract_flags](#list_extract_flags)
* [list_extract_labelled_keyvalue](#list_extract_labelled_keyvalue)
* [list_extract_labelled_value](#list_extract_labelled_value)
* [list_extract_matches](#list_extract_matches)
* [list_find](#list_find)
* [list_find_any](#list_find_any)
* [list_find_flags](#list_find_flags)
* [list_fold](#list_fold)
* [list_get](#list_get)
* [list_get_labelled_value](#list_get_labelled_value)
* [list_get_lean](#list_get_lean)
* [list_intersect](#list_intersect)
* [list_intersect_args](#list_intersect_args)
* [list_isempty](#list_isempty)
* [list_isinorder](#list_isinorder)
* [list_iterator](#list_iterator)
* [list_iterator_break](#list_iterator_break)
* [list_iterator_next](#list_iterator_next)
* [list_length](#list_length)
* [list_max](#list_max)
* [list_modify](#list_modify)
* [list_normalize_index](#list_normalize_index)
* [list_pad](#list_pad)
* [list_pad_set](#list_pad_set)
* [list_parse_descriptor](#list_parse_descriptor)
* [list_peek_back](#list_peek_back)
* [list_peek_front](#list_peek_front)
* [list_pop_back](#list_pop_back)
* [list_pop_front](#list_pop_front)
* [list_push_back](#list_push_back)
* [list_push_front](#list_push_front)
* [list_regex_match](#list_regex_match)
* [list_regex_match_ignore](#list_regex_match_ignore)
* [list_remove](#list_remove)
* [list_remove_at](#list_remove_at)
* [list_remove_duplicates](#list_remove_duplicates)
* [list_replace_at](#list_replace_at)
* [list_replace_slice](#list_replace_slice)
* [list_reverse](#list_reverse)
* [list_select](#list_select)
* [list_select_property](#list_select_property)
* [list_set_at](#list_set_at)
* [list_slice](#list_slice)
* [list_sort](#list_sort)
* [list_split](#list_split)
* [list_split_at](#list_split_at)
* [list_swap](#list_swap)
* [list_to_map](#list_to_map)
* [list_to_string](#list_to_string)
* [list_union](#list_union)
* [list_unique](#list_unique)
* [list_where](#list_where)
* [list_without_range](#list_without_range)
* [is_range](#is_range)
* [list_range_get](#list_range_get)
* [list_range_indices](#list_range_indices)
* [list_range_partial_write](#list_range_partial_write)
* [list_range_remove](#list_range_remove)
* [list_range_replace](#list_range_replace)
* [list_range_set](#list_range_set)
* [list_range_try_get](#list_range_try_get)
* [range_from_indices](#range_from_indices)
* [range_indices](#range_indices)
* [range_indices_valid](#range_indices_valid)
* [range_instanciate](#range_instanciate)
* [range_parse](#range_parse)
* [range_partial_unpack](#range_partial_unpack)
* [range_simplify](#range_simplify)
* [set_difference](#set_difference)
* [set_isequal](#set_isequal)
* [set_issubset](#set_issubset)
* [structured_list_parse](#structured_list_parse)
* [list_structure_print_help](#list_structure_print_help)

### Function Descriptions

## <a name="encoded_list"></a> `encoded_list`

 creates encoded lists from the specified arguments




## <a name="encoded_list_append"></a> `encoded_list_append`





## <a name="encoded_list_decode"></a> `encoded_list_decode`

 faster




## <a name="encoded_list_get"></a> `encoded_list_get`





## <a name="encoded_list_peek_back"></a> `encoded_list_peek_back`





## <a name="encoded_list_peek_front"></a> `encoded_list_peek_front`





## <a name="encoded_list_pop_back"></a> `encoded_list_pop_back`





## <a name="encoded_list_pop_front"></a> `encoded_list_pop_front`





## <a name="encoded_list_remove_at"></a> `encoded_list_remove_at`





## <a name="encoded_list_remove_item"></a> `encoded_list_remove_item`





## <a name="encoded_list_set"></a> `encoded_list_set`





## <a name="encoded_list_to_cmake_string"></a> `encoded_list_to_cmake_string`





## <a name="is_encoded_list"></a> `is_encoded_list`



 returns true iff the arguments passed are in encoded list format




## <a name="index_range"></a> `index_range`

 returns a list of numbers [ start_index, end_index)
 if start_index equals end_index the list is empty
 if end_index is less than start_index then the indices are in declining order
 ie index_range(5 3) => 5 4
 (do not confuse this function with the `range_` functions)




## <a name="linked_list_insert_after"></a> `linked_list_insert_after`

 `(<linked list> <where: <linked list node> = <linked list>.tail >  <any>... )-><linked list node>`
 
 inserts a new linked list node after `where`. if where is null then the tail of the list is used.
 the arguments passed after where are used as the value of the new node




## <a name="linked_list_insert_before"></a> `linked_list_insert_before`

 `(<linked list> <where: <linked list node> = <linked list>.head)-><linked list node>`

 inserts a new linked list node into the linked list before where and returns it.




## <a name="linked_list_new"></a> `linked_list_new`

 `()-><linked list>`
 
 creates a new linked list 
 
 ```
 <linked list node> ::= <null> | {
   head: <linked list node>|<null>
   tail: <linekd list node>|<null>
 }
 ```




## <a name="linked_list_node_new"></a> `linked_list_node_new`

 `(<any>...)-><linked list node>`
 
 creates a new linked list node which contains the value specified
 




## <a name="linked_list_peek_back"></a> `linked_list_peek_back`





## <a name="linked_list_peek_front"></a> `linked_list_peek_front`





## <a name="linked_list_pop_back"></a> `linked_list_pop_back`





## <a name="linked_list_pop_front"></a> `linked_list_pop_front`





## <a name="linked_list_push_back"></a> `linked_list_push_back`





## <a name="linked_list_push_front"></a> `linked_list_push_front`





## <a name="linked_list_remove"></a> `linked_list_remove`





## <a name="linked_list_replace"></a> `linked_list_replace`

 `(<linked list> <where:<linked list node>> <any>...)-><linked list node>`
  
 replaces the specified linked list node and returns new node




## <a name="list_after"></a> `list_after`

 `(<list ref> <key:<string>>)-><any ....>`

 returns the elements after the specified key




## <a name="list_all"></a> `list_all`

 `(<list&> <predicate:<[](<any>)->bool>>)-><bool>` 

 returns true iff predicate holds for all elements of `<list>` 
 




## <a name="list_any"></a> `list_any`

 `[](<list&> <predicate:<[](<any>)->bool>)-><bool>`

 returns true if there exists an element in `<list>` for which the `<predicate>` holds




## <a name="list_append"></a> `list_append`

 safe append (can also append empty element)




## <a name="list_at"></a> `list_at`

 

 returns all elements whose index are specfied
 




## <a name="list_before"></a> `list_before`

 `(<list&> <key:<string>>)-><any ....>`

 returns the elements before key




## <a name="list_check_items"></a> `list_check_items`

 `(<list&> <query...>)-><bool>`
  
 `<query> := <value>|'!'<value>|<value>'?'`
 
 * checks to see that every value specified is contained in the list 
 * if the value is preceded by a `!` checks that the value is not in the list
 * if the value is succeeded by a `?` the value may or may not be contained

 returns true if all queries match
 




## <a name="list_combinations"></a> `list_combinations`

 `(<list&...>)-><any...>`

 returns all possible combinations of the specified lists
 e.g.
 ```
 set(range 0 1)
 list_combinations(range range range)
 ans(result)
 assert(${result} EQUALS 000 001 010 011 100 101 110 111)
 ```





## <a name="list_contains"></a> `list_contains`

 `(<list&> <element:<any...>>)-><bool>`

 returns true if list contains every element specified 





## <a name="list_contains_any"></a> `list_contains_any`





## <a name="list_count"></a> `list_count`

 `(<list&> <predicate:<[](<any>)-><bool>>> )-><uint>`

 counts all element for which the predicate holds 




## <a name="list_equal"></a> `list_equal`





## <a name="list_erase"></a> `list_erase`





## <a name="list_erase_slice"></a> `list_erase_slice`





## <a name="list_except"></a> `list_except`





## <a name="list_extract"></a> `list_extract`





## <a name="list_extract_any_flag"></a> `list_extract_any_flag`





## <a name="list_extract_any_labelled_value"></a> `list_extract_any_labelled_value`

 extracts any of the specified labelled values and returns as soon 
 the first labelled value is found
 lst contains its original elements without the labelled value 




## <a name="list_extract_flag"></a> `list_extract_flag`





## <a name="list_extract_flag_name"></a> `list_extract_flag_name`

 extracts a flag from the list if it is found 
 returns the flag itself (usefull for forwarding flags)




## <a name="list_extract_flags"></a> `list_extract_flags`





## <a name="list_extract_labelled_keyvalue"></a> `list_extract_labelled_keyvalue`

 extracts a labelled key value (the label and the value if it exists)




## <a name="list_extract_labelled_value"></a> `list_extract_labelled_value`





## <a name="list_extract_matches"></a> `list_extract_matches`

 `(<&> <regex>...)-><any...>`

 removes all matches from the list and returns them
 sideffect: matches are removed from list




## <a name="list_find"></a> `list_find`





## <a name="list_find_any"></a> `list_find_any`

 returns the index of the one of the specified items
 if no element is found then -1 is returned 
 no guarantee is made on which item's index
 is returned 




## <a name="list_find_flags"></a> `list_find_flags`

 returns a map of all found flags specified as ARGN
  




## <a name="list_fold"></a> `list_fold`





## <a name="list_get"></a> `list_get`

 returns the item at the specified index
 the index is normalized (see list_normalize_index)




## <a name="list_get_labelled_value"></a> `list_get_labelled_value`

 gets the labelled value from the specified list
 set(thelist a b c d)
 list_get_labelled_value(thelist b) -> c




## <a name="list_get_lean"></a> `list_get_lean`

 quickly gets the items from the specified list




## <a name="list_intersect"></a> `list_intersect`





## <a name="list_intersect_args"></a> `list_intersect_args`





## <a name="list_isempty"></a> `list_isempty`





## <a name="list_isinorder"></a> `list_isinorder`





## <a name="list_iterator"></a> `list_iterator`

 instanciates a list_iterator from the specified list




## <a name="list_iterator_break"></a> `list_iterator_break`

 advances the iterator using list_iterator_next 
 and breaks the current loop when the iterator is done




## <a name="list_iterator_next"></a> `list_iterator_next`

 advances the iterator specified 
 and returns true if it is on a valid element (else false)
 sets the fields 
 ${it_ref}.index
 ${it_ref}.length
 ${it_ref}.list_ref
 ${it_ref}.value (only if a valid value exists)




## <a name="list_length"></a> `list_length`

 returns the length of the specified list




## <a name="list_max"></a> `list_max`

 returns the maximum value in the list 
 using the specified comparerer function




## <a name="list_modify"></a> `list_modify`





## <a name="list_normalize_index"></a> `list_normalize_index`





## <a name="list_pad"></a> `list_pad`





## <a name="list_pad_set"></a> `list_pad_set`

 pads the list so that every index is set then applies the specified value




## <a name="list_parse_descriptor"></a> `list_parse_descriptor`





## <a name="list_peek_back"></a> `list_peek_back`

 Returns the last element of a list without modifying it




## <a name="list_peek_front"></a> `list_peek_front`





## <a name="list_pop_back"></a> `list_pop_back`





## <a name="list_pop_front"></a> `list_pop_front`





## <a name="list_push_back"></a> `list_push_back`





## <a name="list_push_front"></a> `list_push_front`





## <a name="list_regex_match"></a> `list_regex_match`

 matches all elements of lst to regex
 all elements in list which match the regex are returned




## <a name="list_regex_match_ignore"></a> `list_regex_match_ignore`

 returns every element of lst that matches any of the given regexes
 and does not match any regex that starts with !




## <a name="list_remove"></a> `list_remove`





## <a name="list_remove_at"></a> `list_remove_at`





## <a name="list_remove_duplicates"></a> `list_remove_duplicates`

 removes duplicates from a list




## <a name="list_replace_at"></a> `list_replace_at`





## <a name="list_replace_slice"></a> `list_replace_slice`

 replaces the specified slice with the specified varargs
 returns the elements which were removed




## <a name="list_reverse"></a> `list_reverse`

 `(<list ref>)-><void>`

 reverses the specified lists elements




## <a name="list_select"></a> `list_select`





## <a name="list_select_property"></a> `list_select_property`





## <a name="list_set_at"></a> `list_set_at`





## <a name="list_slice"></a> `list_slice`





## <a name="list_sort"></a> `list_sort`





## <a name="list_split"></a> `list_split`

 assert allows assertion




## <a name="list_split_at"></a> `list_split_at`

 list_split_at()






## <a name="list_swap"></a> `list_swap`





## <a name="list_to_map"></a> `list_to_map`





## <a name="list_to_string"></a> `list_to_string`





## <a name="list_union"></a> `list_union`





## <a name="list_unique"></a> `list_unique`





## <a name="list_where"></a> `list_where`





## <a name="list_without_range"></a> `list_without_range`





## <a name="is_range"></a> `is_range`








## <a name="list_range_get"></a> `list_range_get`

 returns the elements of the specified list ref which are indexed by specified range




## <a name="list_range_indices"></a> `list_range_indices`

 list_range_indices(<list&> <range ...>)
 returns the indices for the range for the specified list
 e.g. 
 




## <a name="list_range_partial_write"></a> `list_range_partial_write`

 writes the specified varargs to the list
 at the beginning of the specified partial range
 fails if the range is a  multi range
 e.g. 
 set(lstB a b c)
 list_range_partial_write(lstB "[]" 1 2 3)
 -> lst== [a b c 1 2 3]
 list_range_partial_write(lstB "[1]" 1 2 3)
 -> lst == [a 1 2 3 c]
 list_range_partial_write(lstB "[1)" 1 2 3)
 -> lst == [a 1 2 3 b c]




## <a name="list_range_remove"></a> `list_range_remove`

 removes the specified range from the list




## <a name="list_range_replace"></a> `list_range_replace`

 replaces the specified range with the specified arguments
 the varags are taken and fill up the range to replace_count
 e.g. set(list a b c d e) 
 list_range_replace(list "4 0 3:1:-2" 1 2 3 4 5) --> list is equal to  2 4 c 3 1 





## <a name="list_range_set"></a> `list_range_set`

 sets every element included in range to specified value
 




## <a name="list_range_try_get"></a> `list_range_try_get`

 `(<&list> )`

 returns the elements of the specified list ref which are indexed by specified range




## <a name="range_from_indices"></a> `range_from_indices`

 `(<index:<uint>...>)-><instanciated range...>`
 
 returns the best ranges from the specified indices
 e.g range_from_indices(1 2 3) -> [1:3]
     range_from_indices(1 2) -> 1 2
     range_from_indices(1 2 3 4 5 6 7 8 4 3 2 1 9 6 7) -> [1:8] [4:1:-1] 9 6 7




## <a name="range_indices"></a> `range_indices`

 `(<length:<int>> <~range...>)-><index:<uint>...>` 

 returns the list of indices for the specified range
 length may be negative which causes a failure if any anchors are used (`$` or `n`) 
 
 if the length is valid  (`>-1`) only valid indices are returned or failure occurs

 a length of 0 always returns no indices

 **Examples**
 ```
 ```




## <a name="range_indices_valid"></a> `range_indices_valid`

 returns all valid indices for the specified range




## <a name="range_instanciate"></a> `range_instanciate`

 `(<length:<int>> <~range...>)-><instanciated range...>`
 
 instanciates a range.  A uninstanciated range contains anchors
 these are removed when a length is specified (`n`)
 returns a valid range  with no anchors




## <a name="range_parse"></a> `range_parse`

 `(<~range...>)-><range>`

 parses a range string and normalizes it to have the following form:
 `<range> ::= <begin>":"<end>":"<increment>":"<begin inclusivity:<bool>>":"<end inclusivity:<bool>>":"<length>":"<reverse:<bool>>
 these `<range>`s can be used to generate a index list which can in turn be used to address lists.
  
   * a list of `<range>`s is a  `<range>`  
   * `$` the last element 
   * `n` the element after the last element ($+1)
   * `-<n>` a begin or end starting with `-` is transformed into `$-<n>`
   * `"["` `"("` `")"` and `"]"`  signify the inclusivity.  
 




## <a name="range_partial_unpack"></a> `range_partial_unpack`






## <a name="range_simplify"></a> `range_simplify`

 `(<length:<int>> <range...>)-><instanciated range...>`

 tries to simplify the specified range for the given length
 his is done by getting the indices and then getting the range from indices




## <a name="set_difference"></a> `set_difference`

 `(<listA&:<any...> <listB&:<any...>>)-><any..>`
 
 




## <a name="set_isequal"></a> `set_isequal`





## <a name="set_issubset"></a> `set_issubset`





## <a name="structured_list_parse"></a> `structured_list_parse`





## <a name="list_structure_print_help"></a> `list_structure_print_help`








