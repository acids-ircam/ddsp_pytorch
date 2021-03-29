# Process Management



This section how to manage processes and external programs.  Besides replacements for cmake's `execute_process` function this section defines a control system for parallel processes controlled from any cmake.  

## Function List


* [async](#async)
* [await](#await)
* [command_line](#command_line)
* [command_line_args_combine](#command_line_args_combine)
* [command_line_args_escape](#command_line_args_escape)
* [command_line_parse](#command_line_parse)
* [command_line_parse_string](#command_line_parse_string)
* [command_line_to_string](#command_line_to_string)
* [execute](#execute)
* [execute_script](#execute_script)
* [process_execute](#process_execute)
* [process_handle](#process_handle)
* [process_handle_change_state](#process_handle_change_state)
* [process_handle_get](#process_handle_get)
* [process_handle_new](#process_handle_new)
* [process_handle_register](#process_handle_register)
* [process_handles](#process_handles)
* [process_info](#process_info)
* [process_isrunning](#process_isrunning)
* [process_kill](#process_kill)
* [process_list](#process_list)
* [process_refresh_handle](#process_refresh_handle)
* [process_return_code](#process_return_code)
* [process_start](#process_start)
* [](#)
* [process_start_info_new](#process_start_info_new)
* [process_start_script](#process_start_script)
* [process_stderr](#process_stderr)
* [process_stdout](#process_stdout)
* [process_timeout](#process_timeout)
* [process_wait](#process_wait)
* [process_wait_all](#process_wait_all)
* [process_wait_any](#process_wait_any)
* [process_wait_n](#process_wait_n)
* [string_take_commandline_arg](#string_take_commandline_arg)
* [wrap_executable](#wrap_executable)
* [wrap_executable_bare](#wrap_executable_bare)


## Common Definitions

The following definitions are common to the following subsections.  

* `<command>` a path or filename to an executable programm.
* `<process start info> ::= { <command>, <<args:<any ...>>, <cwd:<directory>>  }`  


## <a name="execute"></a> Wrapping and Executing External Programms

Using external applications is more complex than necessary in cmake in my opinion. I tried to make it as easy as possible. Using the convenience of the [filesystem functions](#filesystem) and maps wrapping an external programm and using it as well as pure execution is now very simple as you can see in the following example for git:

### Examples
This is all the code you need to create a function which wraps the git executable.  It uses the [initializer function pattern](#initializer_function). 

```
function(git)
  # initializer block (will only be executed once)
  find_package(Git)
  if(NOT GIT_FOUND)
    message(FATAL_ERROR "missing git")
  endif()
  # function redefinition inside wrap_executable
  wrap_executable(git "${GIT_EXECUTABLE}")
  # delegate to redefinition and return value
  git(${ARGN})
  return_ans()
endfunction() 
```

Another example showing usage of the `execute()` function:

```
find_package(Hg)
set(cmdline --version)
execute({path:$HG_EXECUTABLE, args: $cmdline} --process-handle)
ans(res)
map_get(${res} result)
ans(error)
map_get(${res} output)
ans(stdout)
assert(NOT error) # error code is 0
assert("${stdout}" MATCHES "mercurial") # output contains mercurial
json_print(${res}) # outputs input and output of process

```

### Functions and Datatypes

* `execute(<process start info ?!> [--process-handle|--exit-code]) -> <stdout>|<process info>|<int>` executes the process described by `<process start ish>` and by default fails fatally if return code is not 0.  if `--exit-code` flag is specified `<process info>` is returned and if `<return-code>` is specified the command's return code is returned.  (the second two options will not cause a fatal error)
  * example: `execute("{path:'<command>', args:['a','b']}")`  
* `wrap_executable(<name:string> <command>)`  takes the executable/command and wraps it inside a function called `<name>` it has the same signature as `execute(...)`
* `<process start info?!>` a string which can be converted to a `<process start>` object using the `process_start_info()` function.
* `<process start info>` a map/object containing the following fields
  - `command` command name / path of executable *required*
  - `args` command line arguments to pass along to command, use `string_encode_semicolon` if you want to have an argument with semicolons *optional*
  - `timeout:<int>` *optional* number of seconds to wait before failing
  - `cwd:<unqualified path>` *optional*  by default it is whatever `pwd()` currently returns
* `<process info>` contains all the fields of `<process start>` and additionaly
  - `output:<stdout>`  the output of the command execution. (merged error and stdout streams)
  - `result:<int>` the return code of the execution.  0 normally indicates success.
* `process_start_info(<process start info?!>):<process start info>` creates a valid `<process start info>` object from the input argument.  If the input is invalid the function fails fatally.

## Parallel Processes 

When working with large applications in cmake it can become necessary to work in parallel processes. Since all cmake target systems support multitasking from the command line it is possible to implement cmake functions to control it.  I implemented a 'platform independent' (platform on which either powershell or bash is available) control mechanism for starting, killing, querying and waiting for processes.  The lowlevel functions are platform specific the others are based on the abstraction layer that the provide.   

### Examples

This example starts a script into three separate cmake processes. The program ends when all scripts are done executing.
```
# define a script which counts to 10 and then 
# note that a fresh process means that cmake has not loaded cmakepp
set(script "
foreach(i RANGE 0 10)
  message(\${i})
  execute_process(COMMAND \${CMAKE_COMMAND} -E sleep 1)
endforeach()
message(end)
")

# start each script - fork_script returns without waiting for the process to finish.
# a handle to the created process is returned.
process_start_script("${script}")
ans(handle1)
process_start_script("${script}")
ans(handle2)
process_start_script("${script}")
ans(handle3)

# wait for every process to finish. returns the handles in order in which the process finishes
process_wait_all(${handle1} ${handle2} ${handle3})
ans(res)

## print the process handles in order of completion
json_print(${res})

```

This example shows a more usefull case:  Downloading multiple 'large' files parallely to save time

```

  ## define a function which downloads  
  ## all urls specified to the current dir
  ## returns the path for every downloaded files
  function(download_files_parallel)
    ## get current working dir
    pwd()
    ans(target_dir)

    ## process start loop 
    ## starts a new process for every url to download
    set(handles)
    foreach(url ${ARGN})
      ## start download by creating a cmake script
      process_start_script("
        include(${cmakepp_base_dir}/cmakepp.cmake) # include cmakepp
        download(\"${url}\" \"${target_dir}\")
        ans(result_path)
        message(STATUS ${target_dir}/\${result_path})
        ")
      ans(handle)
      ## store process handle 
      list(APPEND handles ${handle})
    endforeach()

    ## wait for all downloads to finish
    process_wait_all(${handles})

    set(result_paths)
    foreach(handle ${handles})
      ## get process stdout
      process_stdout(${handle})
      ans(output)

      ## remove '-- ' from beginning of output which is
      ## automatically prependend by message(STATUS) 
      string(SUBSTRING "${output}" 3 -1 output)

      ## store returned file path
      list(APPEND result_paths ${output})

    endforeach()

    ## return file paths of downloaded files
    return_ref(result_paths)
  endfunction()


  ## create and goto ./download_dir
  cd("download_dir" --create)

  ## start downloading files in parallel by calling previously defined function
  download_files_parallel(
    http://www.cmake.org/files/v3.0/cmake-3.0.2.tar.gz
    http://www.cmake.org/files/v2.8/cmake-2.8.12.2.tar.gz
  )
  ans(paths)


  ## assert that every the files were downloaded
  foreach(path ${paths})
    assert(EXISTS "${path}")
  endforeach()


```


### Functions and Datatypes
* datatypes
  * `<process handle> ::= { state:<process state> , pid:<process id> }` process handle is a runtime unique map which is used to address a process.  The process handle may contain more properties than specified - only the specified ones are available on all systems - these properties contain values which are implementation specific.
  * `<process info> ::= { }` a map containing verbose information on a proccess. only the specified fields are available on all platforms.  More are available depending on the OS you use. You should not try to use these without examining their origin / validity.
  * `<process state> ::= "running"|"terminated"|"unknown"`
  * `<process id> ::= <string>` a unspecified systemwide unique string which identifies a process (normally a integer)
* platform specific low level functions 
  * `process_start(<process start info?!>):<process handle>` platfrom specific function which starts a process and returns a process handle
  * `process_kill(<process handle?!>)` platform specific function which stops a process.
  * `process_list():<process handle ...>` platform specific function which returns a process handle for all running processes on OS.
  * `process_info(<process handle?!>):<process info>` platform specific function which returns a verbose process info
  * `process_isrunning(<process handle?!>):<bool>` returns true iff process is running. 
* `process_timeout(<n:<seconds>>):<process handle>` starts a process which times out in `<n>` seconds. 
* `process_wait(<process handle~> [--timeout <n:seconds>]):<process handle>` waits for the specified process to finish or the specified timeout to run out. returns null if timeout runs out before process finishes.
* `process_wait_all(<process handle?!...> <[--timeout <n:seconds>] [--quietly]):<process handle ...>` waits for all specified process handles and returns them in the order that they completed.  If the `--timeout <n>` value is specified the function returns as soon as the timeout is reached returning only the process finished up to that point. The function normally prints status messages which can be supressed via the `--quietly` flag.    
* `process_wait_any(<process handle?!...> <?"--timeout" <n:<seconds>>> <?"--quietly">):<?process handle>` waits for any of the specified processes to finish, returning the handle of the first one to finished. If `--timeout <n>` is specified the function will return `null` after `n` seconds if no process completed up to that point in time. You can specify `--quietly` if you want to suppress the status messages. 
* `process_stdout(<process handle~>):<string>` returns all data written to standard output stream of the process specified up to the current point in time
* `process_stderr(<process handle~>):<string>` return all data written to standard error stream of the process specified up to the current point in time
*   `process_return_code(<process handle~>):<int?>` returns nothing or the process return code if the process is finished
*   `process_start_script(<cmake code>):<process handle>` starts a separate cmake process which runs the specified cmake code.

### Inter Process Communication

To communicate with you processes you can use any of the following well known mechanisms

* Environment Variables
  - the started processes have access to you current Environment. So when you call `set(ENV{VAR} value)` before starting a process that process will have read access to the variable `$ENV{VAR}` 
* Command Line Arguments
  - all variables passed to `start_process` will be passed allong
  - Command Line Variables are sometimes problematic as they must be escaped correctly and this does not always happen as expected. So you might want to choose another mechanism to transmit complex data to your process
  - Command Line Variables are limited by their string length depending on you host os.
* Files
  - Files are the easiest and safest way to communicate large amounts of data to another process. If you can try to use file communication
* stderr, stdout
  - The output of a process started with `start_process` becomes available to you when the process ends at latest, You can choose to poll stdout and take data as soon as it is written to the output streams 
* return code
  - the returns code tells you how you process finished and is often enough result information for a process you start
  
### Caveats

* process starting is slow - it can take seconds (it takes 900ms on my machine). The task needs to be a very large one for it to compensate the overhead.
* parallel processes use platform specific functions - It might cause problems on less well tested OSs and some may not be supported.  (currently only platforms with bash or powershell are supported ie Windows and Linux)





## Function Descriptions

## <a name="async"></a> `async`

 async(<callable>(<args...)) -> <process handle>

 executes a callable asynchroniously 

 todo: 
   capture necessary scope vars
   include further files for custom functions     
   environment vars




## <a name="await"></a> `await`





## <a name="command_line"></a> `command_line`





## <a name="command_line_args_combine"></a> `command_line_args_combine`

 combines the list of command line args into a string which separates and escapes them correctly




## <a name="command_line_args_escape"></a> `command_line_args_escape`

 escapes a command line quoting arguments as needed 




## <a name="command_line_parse"></a> `command_line_parse`

 command_line_parse 
 parses the sepcified cmake style  command list which starts with COMMAND 
 or parses a single command line call
 returns a command line object:
 {
   command:<string>,
   args: <string>...
 }




## <a name="command_line_parse_string"></a> `command_line_parse_string`





## <a name="command_line_to_string"></a> `command_line_to_string`





## <a name="execute"></a> `execute`

 `(<process start info> [--process-handle] [--exit-code] [--async] [--silent-fail] [--success-callback <callable>]  [--error-callback <callable>] [--state-changed-callback <callable>])-><process handle>|<exit code>|<stdout>|<null>`

 *options*
 * `--process-handle` 
 * `--exit-code` 
 * `--async` 
 * `--silent-fail` 
 * `--success-callback <callable>[exit_code](<process handle>)` 
 * `--error-callback <callable>[exit_code](<process handle>)` 
 * `--state-changed-callback <callable>[old_state;new_state](<process handle>)` 
 * `--lean`
 *example*
 ```
  execute(cmake -E echo_append hello) -> 'hello'
 ```




## <a name="execute_script"></a> `execute_script`

 `(<cmake code> [--pure] <args...>)-><execute result>`

 equivalent to `execute(...)->...` runs the specified code using `cmake -P`.  
 prepends the current `cmakepp.cmake` to the script  (this default behaviour can be stopped by adding `--pure`)

 all not specified `args` are forwarded to `execute`





## <a name="process_execute"></a> `process_execute`

 `(<process handle>)-><process handle>`

 executes the specified command with the specified arguments in the 
 current working directory
 creates and registers a process handle which is then returned 
 this function accepts arguments as encoded lists. this allows you to include
 arguments which contain semicolon and other special string chars. 
 the process id of processes start with `process_execute` is always -1
 because `CMake`'s `execute_process` does not return it. This is not too much of a problem
 because the process will always be terminated as soon as the function returns
 
 **parameters**
   * `<command>` the executable (may contain spaces) 
   * `<arg...>` the arguments - may be an encoded list 
 **scope**
   * `pwd()` used as the working-directory
 **events** 
   * `on_process_handle_created` global event is emitted when the process_handle is ready
   * `process_handle.on_state_changed`

 **returns**
 ```
 <process handle> ::= {
   pid: "-1"|"0"
     
 }
 ``` 




## <a name="process_handle"></a> `process_handle`

 returns the runtime unique process handle
 information may differ depending on os but the following are the same for any os
 * pid
 * status




## <a name="process_handle_change_state"></a> `process_handle_change_state`





## <a name="process_handle_get"></a> `process_handle_get`





## <a name="process_handle_new"></a> `process_handle_new`

 `(<process start info>)-><process handle>`

 returns a new process handle which has the following layout:
 ```
 <process handle> ::= {
   pid: <pid>  
   start_info: <process start info>
   state: "unknown"|"running"|"terminated"
   stdout: <text>
   stderr: <text>
   exit_code: <integer>|<error string>
   command: <executable>
   command_args: <encoded list>
   on_state_change: <event>[old_state, new_state](${process_handle}) 
 }
 ``` 




## <a name="process_handle_register"></a> `process_handle_register`





## <a name="process_handles"></a> `process_handles`

 transforms a list of <process handle?!> into a list of <process handle>  




## <a name="process_info"></a> `process_info`

 process_info(<process handle?!>): <process info>
 returns information on the specified process handle




## <a name="process_isrunning"></a> `process_isrunning`

 returns true iff the process identified by <handlish> is running




## <a name="process_kill"></a> `process_kill`





## <a name="process_list"></a> `process_list`

 returns a list of <process info> containing all processes currently running on os
 process_list():<process info>...




## <a name="process_refresh_handle"></a> `process_refresh_handle`

 refreshes the fields of the process handle
 returns true if the process is still running false otherwise
 this is the only function which is allowed to change the state of a process handle




## <a name="process_return_code"></a> `process_return_code`

 returns the <return_code> for the specified process handle
 if process is not finished the result is empty




## <a name="process_start"></a> `process_start`

 starts a process and returns a handle which can be used to controll it.  





## <a name="process_start_info_new"></a> `process_start_info_new`

 `(<command string>|<object> [TIMEOUT <n:int>] [WORKING_DIRECTORY ~<path>] [--passthru])-><process start info>` 
 `<command string> ::= "COMMAND"? <command> <arg...>` 

 creates a new process_start_info with the following fields
 ```
 <process start info> ::= {
   command: <executable> 
   command_arguments: <encoded list>
   working_directory: <directory>
   timeout: <n>
   passthru: true|false
 }
 ```
 the syntax of the process_start_info_new is equivalent to cmake's built in `execute_process` command.


 the command needs to point to an executable,  preferrably a fully qualified path 
 the command_arguments contain an encoded list.  you can specify any argument in the execute function - they will be passed along as you write them  (this deserves extra appreciation because doing so in cmake is hard)
 If the flags you want to pass along to the process conflict with the flags of the process function you can always encode them yourself and pass them along as an encoded list


 *example*
  * `process_start_info_new(COMMAND cmake -E echo "asd bsd" csd) -> {
 "command":"cmake",
 "command_arguments":[
  "-E",
  "echo",
  "asd;bsd"
 ],
 "working_directory":"/home/anotherfoxguy/Documents/Github/cmakepp/cmake/process",
 "timeout":"-1",
 "passthru":false
}` 




## <a name="process_start_script"></a> `process_start_script`

 shorthand to fork a cmake script




## <a name="process_stderr"></a> `process_stderr`

 returns the current error output
 This can change until the process is finished




## <a name="process_stdout"></a> `process_stdout`

 returns the current stdout of a <process handle>
 this changes until the process is ove




## <a name="process_timeout"></a> `process_timeout`

 returns a <process handle> to a process that runs for n seconds




## <a name="process_wait"></a> `process_wait`

 blocks until given process has terminated
 returns nothing if the process does not exist - is deleted etc
 updates and returns the process_handle
 if a timeout greater 0 the function will return nothing if the timeout is reached
 process_wait(<process handle> <?--timeout:<seconds>>)




## <a name="process_wait_all"></a> `process_wait_all`

 `(<handles: <process handle...>>  [--timeout <seconds>] [--idle-callback <callable>] [--task-complete-callback <callable>] )`

 waits for all specified <process handles> to finish returns them in the order
 in which they completed

 `--timeout <n>`    if value is specified the function will return all 
                    finished processes after n seconds

 `--idle-callback <callable>`   
                    if value is specified it will be called at least once
                    and between every query if a task is still running 


 `--task-complete-callback <callable>`
                    if value is specified it will be called whenever a 
                    task completes.

 *Example*
 `process_wait_all(${handle1} ${handle1}  --task-complete-callback "[](handle)message(FORMAT '{handle.pid}')")`
 prints the process id to the console whenver a process finishes





## <a name="process_wait_any"></a> `process_wait_any`

 waits until any of the specified handles stops running
 returns the handle of that process
 if --timeout <n> is specified function will return nothing after n seconds




## <a name="process_wait_n"></a> `process_wait_n`

 `(<n:<int>|"*"> <process handle>... [--idle-callback:<callable>])-><process handle>...`

 waits for at least <n> processes to complete 

 returns: 
  * at least `n` terminated processes
 
 arguments: 
 * `n` an integer the number of processes to return (lower bound) if `n` is clamped to the number of processes. if `n` is * it is replaced with the number of processes 
 * `--idle-callback` is called after every time a processes state was polled. It is guaranteed to be called once per process handle. it has access to the following scope variables
    * `terminated_count` number of terminated processes
    * `running_count` number of running processes
    * `wait_time` time that was waited
    * `wait_counter` number of times the waiting loop iterated
    * `running_processes` list of running processes 
    * `current_process` the current process being polled
    * `is_running` the running state of the current process
    * `terminated_processes` the list of terminated processes





## <a name="string_take_commandline_arg"></a> `string_take_commandline_arg`





## <a name="wrap_executable"></a> `wrap_executable`





## <a name="wrap_executable_bare"></a> `wrap_executable_bare`

 a fast wrapper for the specified executable
 this should be used for executables that are called often
 and do not need to run async






