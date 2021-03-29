## Parsing and handling semantic versions


A `semantic version` gives a specific meaning to it components. It allows software engineers to quickly determine weather versioned components of their software can be updated or if breaking changes could occur.  

On [semver.org](http://semver.org/) Tom Preston-Werner defines Semantic Versioning 2.0.0 which is recapped in the following.

A `semantic version` is defined by is a version string which is in the following format whose components have a special semantic meaning in regard to the versioned subject:
```
<version number> ::= (0|[1-9][0-9]*)
<version tag> ::= [a-zA-Z0-9-]+
<major> ::= <version number>
<minor> ::= <version number>
<patch> ::= <version number>
<prerelease> ::= <version tag>[.<version tag>]*
<metadata> ::= <version tag>[.<version tag>]*
<semantic_version> ::= <major>.<minor>.<patch>[-<prerelease>][+<metadata>]

```
## Examples

* `1.3.42-alpha.0+build-4902.nightly`
* `4.2.1`
* `0.0.0`

## Description

A `version number` may be any `0` or any positive integer. It may not have leading `0`s. 

`major` is a `version number`.

A `version tag` non-empty alphanumeric string (also allowed: hyphen `-`)

`prerelease` is a period `.` separated list of a `version tag`s. A version with `prerelease` is always of a lower order than a the same version without `prerelease`. 

`<metadata>` is a list of period `.` separated `version tag`s with no meaning to the `semantic version`. It can be considered as user data.

## Semantics

`major` is a `version number`.  `0` signifies that the public interface (API) of a package is not yet defined. The first major version `1` defines the public interface for a package. If the `major` version changes backwards incompatible changes have occured.

`minor` is a `version number` which signifies a backwards compatible change in the public interface of a package.  Updating a package to a new minor version MAY NOT break the dependee.

`patch` is a `version number`  which signifies a change in the internals of a package. ie bugfixes, inner enhancements. 

A `version number` SHOULD be incremented by `1` and `version number` the lower `version number`s are reset to zero. Incrementing a version with prerelease has to be done manually e.g.

* increment `major` `1.23.1` => `2.0.0` 
* increment `minor` `1.23.1` => `1.24.0`
* increment `patch` `1.23.1` => `1.23.2`

## Semantic Version Object

The sematic version object is the following map:

```
{
    "string":"<major>.<minor>.<patch>-<prerelease>+<metadata>"
    "numbers":"<major>.<minor>.<patch>",
    "major":"<major>",
    "minor":"<minor>",
    "patch":"<patch>",
    "prerelease":"<prerelease>",
    "metadata":"<metadata>",
    "tags":[
        <version tag>,
        <version tag>,
        ...
    ],
    "metadatas":[
        <version tag>,
        <version tag>,
        ...
    ]

}
```
## Constraining Versions

A `version constraint` constrains the set of all version to a subset.

```
<version operator> ::= >=|<=|>|<|=|~|! 
<version constraint> ::= <version constraint>"|"<version constraint> 
<version constraint> ::= <version constraint>","<version constraint>
<version constraint> ::= <version operator><lazy version>
```

* `<version constraint> ::= <- <package_version_constraint> , <package_version_constring>` can be AND combined. `,` is the and operator and it has precedence before 
* `<package_version_constraint> <- <package_version_constraint> | <package_version_constring>`: `<package_version_constraint>`s can be or combined
* `<package_version_constraint_operator><package_version>`
* `<package_version_constraint_operator><lazy_version>`
* `<package_version>` -> `~<package_version>`
* `<lazy_version>`
* `<lazy_version>` is a `<package_version>` which may  miss elements. These elements are ie `1` would correspond to `1.0.0`
* a `<package_version_constraint_operator>` can be one of the following
    - `=` `<package_version>`  equals the specified version exactly
    - `<` `<package_version>` is less up to date then specified version
    - `>` `<package_version>` is greater than specified version
    - `>=` `<package_version>` is greater or equal to specified version evaluates to `><package_version> | =<package_version>`
    - `<=` `package_version` is less or equal than specified version evaluates to `<<package_version> | =<package_version>`
    - `~` 





## Lazy Version

A `lazy version` a less strict formulation of a `sematic version` 
```
<lazy version> ::= [whitespace]<<sematic_version>|v<lazy_version>|<major>|<major>.<minor>|"">[whitespace]
```

A lazy version allows whitesspace and omission of `minor` and `patch` numbers. It also allows a preceding `v` as is common in many versioning schemes.

A `lazy version` can be normalized to a strict `semantic version` by removing any whitespace around and in the version as well as the leading `v`, and filling up missing `major` and `minor` and `patch` version components with `0`. Normalizing an empty (or pure whitespace) string results in version `0.0.0`

*Examples*
* `v1.3` => `1.3.0`
* `v1-alpha` => `1.0.0-alpha`
* `v1.3-alpha` => `1.3.0-alpha`
* `1` => `1.0.0`
* `  1    ` => `1.0.0`
* `     `=> `0.0.0`
* 
## Functions

The following functions are usable for semantic versioning.

* `semver(<lazy_version>)` parses a string or a semantic version object and returns a `<semantic version object>`
  - `semver(1.0)` => `{major:1,minor:0,patch:0}`
  - `semver(2-alpha+build3.linux)` => `{major:2,minor:0,patch:0,prerelease:['alpha'],metadata:['build3','linux']}`
  - `semver(2.3.1-beta.3+tobi.katha)` => `{major:2,minor:3,patch:1,prerelease:['beta','3'],metadata:['tobi','katha']}`
* `semver_compare(<lhs:semverish> <rhs:semverish>)` compares two semantiv versions.
  - returns `-1` if left is more up to date
  - returns `1` if right is more up to date
  - returns `0` if they are the same
* `semver_higher(<lhs:semverish> <rhs:semverish>)` returns the semantic version which is higher.
* `semver_gt(<lhs:semverish> <rhs:semverish>)` returns true iff left semver is greater than right semver
* `semver_cosntraint_evaluate(<version constraint> <lazy_version>)` returns true if `<lazy_version>` satisfies `<version cosntraint>`
  - `semver_constraint_evaluate("=0.0.1" "0.0.1")` -> true
  - `semver_constraint_evaluate("=0.0.1" "0.0.2")` -> false
    - `semver_constraint_evaluate("!0.0.1" "0.0.1")` -> false
    - `semver_constraint_evaluate("!0.0.1" "0.0.2")` -> true
    - `semver_constraint_evaluate(">0.0.1" "0.0.2")` -> true
    - `semver_constraint_evaluate(">0.0.1" "0.0.1")` -> false
    - `semver_constraint_evaluate("<0.0.1" "0.0.0")` -> true
    - `semver_constraint_evaluate("<0.0.1" "0.0.1")` -> false
    - `semver_constraint_evaluate("<=3,>2" "3.0.0")` -> true
    - `semver_constraint_evaluate("<=3,>=2" "2.0.0")` -> true


## Caveats

* parsing, constraining and comparing semvers is slow. Do not use too much (you can  compile a semver constraint if it is to be evaluated agains many versions which helps a little with performance issues).  

