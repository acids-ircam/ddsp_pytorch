##  Date/Time

I have provided you with a functions which returns a datetime object to get the current date and time on all OSs including windows. It uses file(TIMESTAMP) internally so the resolution is 1s.  It would be possible to enhance this functionality to included milliseconds however that is more system dependent and therefore a decieded against it.  

`datetime()` currently only returns the local time. extending it to return UTC would be easy but I have not yet needed it

In the future date time arithmetic might be added

## Functions

* `datetime()` returns the current date time as a `<datetime object>`
* `<datetime object>` an object containing the following fields: `yyyy` `MM` `dd` `hh` `mm` `ss`