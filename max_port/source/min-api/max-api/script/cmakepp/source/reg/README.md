## Windows Registry


Even though cmake and cmakepp are platform independent working with the windows registry is sometimes import/ e.g. setting environment variables. The cmake interface for manipulating registry values is not very nice (`cmake -E delete_regv` `write_regv`, `get_filename_component(result [HKEY_CURRENT_USER/Environment/Path] ABSOLUTE CACHE)` ) and hard to work with. Therefore I implemented a wrapper for the windows registry command line tool [REG](http://technet.microsoft.com/en-us/library/cc732643.aspx) and called it `reg()` it has the same call signature as `REG` with a minor difference: what is `reg add HKCU/Environment /v MyVar /f /d myval` is written `reg(add HKCU/Environment /v /MyVar /f /d myval)`. See also [wrap_executable](#executable)


## Availables Functions


Using this command I have added convinience functions for manipulating registry keys and values

* `reg()` access to REG command line tool under windows (fails on other OSs)
* `reg_write_value(key value_name value)` writes a registry value (overwrites if it exists)
* `reg_read_value(key value_name)` returns the value of a registry value
* `reg_query_values(key)` returns a map containing all values of a specific registry key
* `reg_append_value(key value_name [args...])` append the specified values to the registries value
* `reg_prepend_value(key value_name [args...])` prepends the specified values to the registries value
* `reg_append_if_not_exists(key value_name [args ...]) appends the specifeid values to the registries value if they are not already part of it, returns only the values which were appended as result
* `reg_remove_value(key value_name [args ...])` removes the specified values from the registries value
* `reg_contains_value(key value_name)` returns true iff the registry contains the specified value
* `reg_query(key)` returns a list of `<reg_entry>` objects which describe found values
* `<reg_entry>` is a object with the fields key, value_name, value, type which describes a registry entry



## Using windows registry functions example


```
set(kv HKCU/Environment testValue1)

## read/write
reg_write_value(${kv} "b;c")
reg_read_value(${kv})
ans(res)
assert(EQUALS ${res} b c)

## append
reg_append_value(${kv} "d")
reg_read_value(${kv})
ans(res)
assert(EQUALS ${res} b c d)

## prepend
reg_prepend_value(${kv} "a")
reg_read_value(${kv})
ans(res)
assert(EQUALS ${res} a b c d)


## append if not exists
reg_append_if_not_exists(${kv} b c e f)
ans(res)
assert(res)
assert(EQUALS ${res} e f)
reg_read_value(${kv})
ans(res)
assert(EQUALS ${res} a b c d e f)


## remove
reg_remove_value(${kv} b d f)
reg_read_value(${kv})
ans(res)
assert(EQUALS ${res} a c e)


## contains
reg_contains_value(${kv} e)  
ans(res)
assert(res)


## read key
reg_query_values(HKCU/Environment)
ans(res)
json_print(${res})
assert(EQUALS DEREF {res.testValue1} a c e)
```
