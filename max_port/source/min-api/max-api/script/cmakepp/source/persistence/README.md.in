## User Data

User Data is usefull for reading and writing configuration per user.  It is available for all cmake execution and can be undestood as an extra variable scope. It however allows maps which help structure data more clearly.  User Data is stored in the users home directory (see `home_dir`) where a folder called `.cmakepp` inside are files in a quickmap format which can be edited in an editor of choice besides being managed by the following functions.  User Data is always read and persisted directly (which is slower but makes the system more consistent)

## Functions and Datatypes

* `<identifier>`  a string
* `user_data_get(<id:<identifier>> [<nav:<navigation expression>>|"."|""]):<any>` returns the user data for the specified identifier, if a navigation expression is specified the userdata map will be navigated to the specified map path and the data is returned (or null if the data does not exist). 
* `user_data_set(<id:<identifier>> <<nav:<navigation expression>>|"."|""|> [<data:any ...>]):<qualified path>` sets the user data identified by id and navigated to by  navigation
* `user_data_dir():<qualified path>` returns the path where the userdata is stored: `$HOME_DIR/.cmakepp`
* `user_data_ids():<identifier ...>` returns a set of identifiers where user data was stored
* `user_data_clear(<"--all"^<id:<identifier>>>):<void>` if `--all` is specified all user data is removed. (use with caution) else only the user data identified by `<id>` is removed
* `user_data_read(<id:<identifier>>):<any>` deserializes the user data identified by id and returns it (`user_data_get` and `user_data_set` are based on this function)
* `user_data_write(<id:<identifier>> [<data:<any> ...>]):<qualified path>` serializes and persists the specified data and associates it with `<id>`
* `user_data_path(<id:<identifier>> ):<qualified path>` returns the filename under which the user data identified by `<id>` is located

## Example

```

## store user data during cmake script execution/configuration/generation steps
## this call creates and stores data in the users home directory/.cmakepp
user_data_set(myuserdata configoptions.configvalue "my value" 34)


## any other file executed afterwards
user_data_get(myuserdata)
ans(res)

json_print(${res}) 

## outputs the following
# {
#   configoptions:{
#     configvalue:["my value",34]
#   }
# }

```
