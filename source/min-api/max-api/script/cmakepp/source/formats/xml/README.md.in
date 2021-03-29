##  Naive Xml Deserialization

Xml Deserialization is a complex subject in CMake.  I have currently implemented a single naive implementation based on regular expressions which *does not* allow recursive nodes (ie nodes with the same tag being child of one another).



### Functions

* `xml_node(<name> <value> <attributes:object>)->{ tag:<name>, value:<value>, attrs:{<key>:<value>}}`  creates a naive xml node representation.
* `xml_parse_nodes(<xml:string> <tag:string>)-> list of xml_nodes`  this function looks for the specified tag in the string (matching every instance(no recursive tags)) it parses the found nodes attributes and value (innerXml). You can then use `nav()`, `map_get()` etc functions to parse the nodes

