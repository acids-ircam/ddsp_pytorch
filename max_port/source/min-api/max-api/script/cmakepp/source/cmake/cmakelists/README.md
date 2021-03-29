# CMakeLists Reflection


Based on the cmake reflection functions I developed I added clases which help you to analyze and modify CMakeLists files without changing custom modifications.  All changes done by these functions are based on token manipulation. 




## Function List


* [cmakelists_open](#cmakelists_open)
* [cmakelists_close](#cmakelists_close)
* [cmakelists_cli](#cmakelists_cli)
* [cmakelists_close](#cmakelists_close)
* [cmakelists_new](#cmakelists_new)
* [cmakelists_open](#cmakelists_open)
* [cmakelists_paths](#cmakelists_paths)
* [cmakelists_serialize](#cmakelists_serialize)
* [cmakelists_target](#cmakelists_target)
* [cmakelists_target_update](#cmakelists_target_update)
* [cmakelists_targets](#cmakelists_targets)
* [cmakelists_variable](#cmakelists_variable)
* [cml](#cml)

## Function Descriptions

## <a name="cmakelists_open"></a> `cmakelists_open`

 `([<path>])-><cmakelists>|<null>`

 opens a the closests cmakelists file (anchor file) found in current or parent directory
 returns nothing if no cmakelists file is found. 




## <a name="cmakelists_close"></a> `cmakelists_close`

 `(<cmakelists>)-> <bool>`

 closes the specified cmakelists file.  This causes it to be written to its path
 returns true on success




## <a name="cmakelists_cli"></a> `cmakelists_cli`

 `([-v])-><any...>`

 the comand line interface to cmakelists.  tries to find the CMakelists.txt in current or parent directories
 if init is specified a new cmakelists file is created in the current directory
 *flags*:
  * 
 *commands*:
  * `init` saves an initial cmake file at the current location
  * `target <target name> <target command> | "add" <target name>` target commands:
    * `add` adds the specified target to the end of the cmakelists file
    * `sources "append"|"set"|"remove" <glob expression>...` adds appends,sets, removes the source files specified by glob expressions to the specified target
    * `includes "append"|"set"|"remove" <path>....` adds the specified directories to the target_include_directories of the specified target
    * `links "append"|"set"|"remove" <target name>...` adds the specified target names to the target_link_libraries of the specified target
    * `type <target type>` sets the type of the specified target to the specified target type
    * `rename <target name>` renames the specified target 
 
 `<target type> ::= "library"|"executable"|"custom_target"|"test"`  




## <a name="cmakelists_close"></a> `cmakelists_close`

 `(<cmakelists>)-> <bool>`

 closes the specified cmakelists file.  This causes it to be written to its path
 returns true on success




## <a name="cmakelists_new"></a> `cmakelists_new`





## <a name="cmakelists_open"></a> `cmakelists_open`

 `([<path>])-><cmakelists>|<null>`

 opens a the closests cmakelists file (anchor file) found in current or parent directory
 returns nothing if no cmakelists file is found. 




## <a name="cmakelists_paths"></a> `cmakelists_paths`

 `(<cmakelists> <file>... [--glob] )-> <relative path>...`

 qualifies the paths relative to the cmakelists directory 
 if `--glob` is specified then the `<file>...` will be treated
 as glob expressions




## <a name="cmakelists_serialize"></a> `cmakelists_serialize`

 `(<cmakelists>)-> <cmake code>`

 serializes the specified cmakelists into its textual representation.




## <a name="cmakelists_target"></a> `cmakelists_target`

 `(<cmakelists> <target:<target name regex>|<cmake target>)-><cmake target> v {target_invocations: <target invocations>}`

 tries to find the single target identified by the regex and returns it. 
 
 ```
 <target> ::= {
    target_name: <string>
    target_type: "library"|"executable"|"test"|"custom_target"|...
    target_source_files
    target_include_directories
    target_link_libraries
    target_compile_definitions
    target_compile_options
 }
 ```




## <a name="cmakelists_target_update"></a> `cmakelists_target_update`

 `(<cmakelists> <cmake target>)-><bool>`
 
 updates the cmakelists tokens to reflect changes in the target
  extrac functions
 




## <a name="cmakelists_targets"></a> `cmakelists_targets`

 `(target_name:<regex>)-><cmake target>`

 returns all targets whose name match the specified regular expression




## <a name="cmakelists_variable"></a> `cmakelists_variable`

 `(<cmakelists> <variable path>)-><any>...`
 
 see list_modify
 modifies a variable returns the value of the variable




## <a name="cml"></a> `cml`

 `(...)->...`
 
 wrapper for cmakelists_cli







