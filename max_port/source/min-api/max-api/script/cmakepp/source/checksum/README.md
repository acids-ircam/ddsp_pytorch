# Creating Checksums

`CMake` provides you with two hashing functions: `string(<hash-algorithm>)` and `file(<hash-algorithm>)`.  These work just fine but I reformed them to work well with `cmakepp` and extended the functionality: 




* [checksum_dir](#checksum_dir)
* [checksum_file](#checksum_file)
* [checksum_files](#checksum_files)
* [checksum_glob_ignore](#checksum_glob_ignore)
* [checksum_layout](#checksum_layout)
* [checksum_object](#checksum_object)
* [checksum_string](#checksum_string)
* [content_dir_check](#content_dir_check)
* [content_dir_update](#content_dir_update)

### Function Details

## <a name="checksum_dir"></a> `checksum_dir`

 `(<direcotry> [--algorthm <checksum algorithm> = "MD5"])-><checksum>`

 calculates the checksum for the specified directory 
 just like checksum_layout however also factors in the file's contents
 




## <a name="checksum_file"></a> `checksum_file`

 `(<file> [--algorithm <checksum algorithm> = "MD5"])-><checksum>`

 calculates the checksum for the specified file delegates the
 call to `CMake`'s file(<algorithm>) function
 




## <a name="checksum_files"></a> `checksum_files`

 `(<base dir> <file...>)-><checksum>`

 create a checksum from specified files relative to <dir>
 the checksum is influenced by the files relative paths 
 and the file content 
 




## <a name="checksum_glob_ignore"></a> `checksum_glob_ignore`

 `(<glob ignore expressions...> [--algorithm <hash algorithm> = "MD5"])-><checksum>`
 
 calculates the checksum for the specified glob ignore expressIONS
 uses checksum_files internally. the checksum is unique to file content
 and relative file structure
 




## <a name="checksum_layout"></a> `checksum_layout`

 `(<directory> [--algorithm <hash algorithm> "MD5"])-><checksum>`
 
 this method generates the checksum for the specified directory
 it is done by taking every file's relative path into consideration
 and generating the hash.  The file's content does not influence the hash
 




## <a name="checksum_object"></a> `checksum_object`

 `(<any> [--algorithm <hash algorithm> = "MD5"])-><checksum>`

 this function takes any value and generates its hash
 the difference to string hash is that it serializes the specified object 
 which lets you create the hash for the whoile object graph.  
 




## <a name="checksum_string"></a> `checksum_string`

 `(<string> [--algorithm <hash algorithm> = MD5])-><checksum>`
 `<hash algorithm> ::= "MD5"|"SHA1"|"SHA224"|"SHA256"|"SHA384"|"SHA512"`

 this function takes any string and computes the hash value of it using the 
 hash algorithm specified (which defaults to  MD5)
 returns the checksum
 




## <a name="content_dir_check"></a> `content_dir_check`





## <a name="content_dir_update"></a> `content_dir_update`






