## Filesystem

I have always been a bit confused when working with cmake's file functions and the logic behind paths (sometimes they are found sometimes they are not...) For ease of use I reimplemented a own path managing system which behaves very similar to powershell and bash (see [ss64.com](http://ss64.com/bash/)) and is compatible to CMake's understanding of paths. It is based around a global path stack and path qualification. All of my functions which work with paths use this system. To better show you what I mean I created the following example:

```
# as soon as you include `cmakepp.cmake` the current directory is set to 
# "${CMAKE_SOURCE_DIR}" which is the directory from which you script file 
# is called in script mode (`cmake -P`) or the directory of the root 
# `CMakeLists.txt` file in configure and build steps.
pwd() # returns the current dir
ans(path)

assert("${path}" STREQUAL "${CMAKE_SOURCE_DIR}")


pushd("dir1" --create) # goto  ${CMAKE_SOURCE_DIR}/dir1; Create if not exists
ans(path)
assert("${path}" STREQUAL "${CMAKE_SOURCE_DIR}/dir1")

fwrite("README.md" "This is the readme file.") # creates the file README.md in dir1
assert(EXISTS "${CMAKE_SOURCE_DIR}/dir1/README.md") 


pushd(dir2 --create) # goto ${CMAKE_SOURCE_DIR}/dir1/dir2 and create it if it does not exist
fwrite("README2.md" "This is another readme file")

cd(../..) # use relative path specifiers to navigate path stack
ans(path)

assert(${path} STREQUAL "${CMAKE_SOURCE_DIR}") # up up -> we are where we started

popd() # path stack is popped. path before was ${CMAKE_SOURCE_DIR}/dir1
ans(path)

assert(${path} STREQUAL "${CMAKE_SOURCE_DIR}/dir1")


mkdir("dir3")
cd(dir3)
# current dir is now ${CMAKE_SOURCE_DIR}/dir1/dir3

# execute() uses the current pwd() as the working dir so the following
# clones the cmakepp repo into ${CMAKE_SOURCE_DIR}/dir1/dir3
git(clone https://github.com/AnotherFoxGuy/cmakepp.git ".")


# remove all files and folders
rm(.)


popd() # pwd is now ${CMAKE_SOURCE_DIR} again and stack is empty

```


## Functions and datatypes

* `<directory> ::= <path|qualifies to an existing directory>` 
* `<file> ::= <path|qualifies to an existing file>`
* `<windows path>`  a windows path possibly with and possibly with drive name `C:\Users\Tobi\README.md`
* `<relative path>` a simple relative path '../dir2/./test.txt'
* `<home path>` a path starting with a tilde `~` which is resolved to the users home directory (under windows and posix)
* `<qualified path>` a fully qualified path depending on OS it only contains forward slashes and is cmake's `get_filename_component(result "${input} REAL_PATH)` returns. All symlinks are resolved. It is absolute
* `<unqualified path> ::= <windows path>|<relative path>|<home path>|<qualified path>` 
* `<path> ::= <unqualified path>`
* `path(<unqualified path>)-><qualified path>` qualifies a path and returns it.  if path is relative (with no drive letter under windows or no initial / on unix) it will be qualified with the current directory `pwd()`
* `pwd()-> <qualified path>` returns the top of the path stack. relative paths are relative to `pwd()`
* `cd(<unqualified> [--create]) -> <qualified path>` changes the top of the path stack.  returns the `<qualified path>` corresonding to input. if `--create` is specified the directory will be created if it does not exist. if `cd()` is navigated towards a non existing directory and `--create` is not specified it will cause a `FATAL_ERROR`
* `pushd(<unqualified path> [--create]) -> <qualified path>` works the same `cd()` except that it pushes the top of the path stack down instead of replacing it
* `popd()-><qualified path>` removes the top of the path stack and returns the new top path
* `dirs()-> <qualified path>[]` returns all paths in the path stack from bottom to top
* file functions
  - `fread(<unqualified path>)-><string>` returns the contents of the specified file
  - `lines(<unqualified path>)-><string>[]` returns the contents of the specified file in a list of lines
  - `download(<uri> [<target:unqualified path>] [--progress])` downloads the file to target, if target is an existing directory the downloaded filename will be extracted from uri else path is treated as the target filepath
  - `fappend(<unqualified path> <content:string>)->void` appends the specified content to the target file
  - `fwrite(<unqualified path> <content:string>)->void` writes the content to the target file (overwriting it)
  - `parent_dir(<unqualified path>)-><qualified path>` returns the parent directory of the specified path
  - `ftime(<unqualified path>)-><timestampstring>` returns the timestamp string for the specified path yyyy-MM-ddThh:mm:ss
  - `ls([<unqualified path>])-><qualified path>[]` returns files and subfolders of specified path
  - `mkdir(<unqualified path>)-><qualfied path>` creates the specified dir and returns its qualified path
  - `mkdirs(<unqualified path>...)-><qualified path>[]` creates all of the directories specified
  - `mktemp([<unqualified path>])-><qualified path>` creates a temporary directory optionally you can specify where this directory is created (by default it is created in TMP_DIR)
  - `mv(<sourcefile> <targetfile>|[<sourcefile> ...] <existing targetdir>)->void` moves the specifeid path to the specified target if last argument is an existing directory all previous files will be moved there else only two arguments are allowed
  - `paths([<unqualified path> ...])-><qualified path>[]` returns the qualified path for every unqualified path received as input
  - `touch(<unqualified path> [--nocreate])-><qualified path>` touches the specified file creating it if it does not exist. if `--nocreate` is specified the file will not be created if it does not exist. the qualified path for the specified file is returned
  - `home_dir()-><qualified path>` returns the users home directory
  - `home_path(<relative path>)-><qualified path>` returns fully qualified  path relative to the user's home directory
  - ... (more functions are coming whenver they are needed)
