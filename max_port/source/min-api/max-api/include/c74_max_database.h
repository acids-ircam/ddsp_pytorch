/// @file
///	@ingroup 	maxapi
///	@copyright	Copyright 2018 The Max-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74 {
namespace max {

    /**	A database object.
        Use db_open() and db_close() to create and free database objects.
        @ingroup	database	*/
    typedef t_object t_database;


    /**	A database result object.
        This is what the database object returns when a query is executed.
        @ingroup	database	*/
    typedef t_object t_db_result;


    /**	A database view object.
        A database view wraps a query and a result for a given database, and is always updated and in-sync with the database.
        @ingroup	database	*/
    typedef t_object t_db_view;


    BEGIN_USING_C_LINKAGE


    /**	Create an instance of a database.
        @ingroup	database
        @param		dbname		The name of the database.
        @param		fullpath	If a database with this dbname is not already open,
                                this will specify a full path to the location where the database is stored on disk.
                                If NULL is passed for this argument, the database will reside in memory only.
                                The path should be formatted as a Max style path.
        @param		db			The address of a #t_database pointer that will be set to point to the new database instance.
                                If the pointer is not NULL, then it will be treated as a pre-existing database instance
                                and thus will be freed.
        @return					An error code.		*/
    t_max_err db_open(t_symbol* dbname, const char* fullpath, t_database** db);

    enum t_db_open_flags {
        DB_OPEN_FLAGS_NONE = 0,
        DB_OPEN_FLAGS_READONLY = 0x01
    };

    /**	Create an instance of a database.
        @ingroup	database
        @param		dbname		The name of the database.
        @param		fullpath	If a database with this dbname is not already open,
                                this will specify a full path to the location where the database is stored on disk.
                                If NULL is passed for this argument, the database will reside in memory only.
                                The path should be formatted as a Max style path.
        @param		db			The address of a #t_database pointer that will be set to point to the new database instance.
                                If the pointer is not NULL, then it will be treated as a pre-existing database instance
                                and thus will be freed.
        @param		flags		Any flags to be passed to the database backend while opening the db. At this time,
                                DB_OPEN_FLAGS_READONLY (0x01) is the only flag available.
        @return					An error code.		*/
    t_max_err db_open_ext(t_symbol* dbname, const char* fullpath, t_database** db, long flags);


    /**	Close an open database.
        @ingroup	database
        @param		db			The address of the #t_database pointer for your database instance.
                                The pointer will be freed and set NULL upon return.
        @return					An error code.		*/
    t_max_err db_close(t_database** db);


    /**	Execute a SQL query on the database.
        @ingroup	database
        @param		db			The #t_database pointer for your database instance.
        @param		dbresult	The address of a #t_db_result pointer.
                                If the pointer is passed-in set to NULL then a new dbresult will be created.
                                If the pointer is not NULL then it is assumed to be a valid dbresult, which will be filled in with the query results.
                                When you are done with the dbresult you should free it with object_free().
        @param		sql			A C-string containing a valid SQL query, possibly with sprintf() formatting codes.
        @param		...			If an sprintf() formatting codes are used in the sql string, these values will be interpolated into the sql string.
        @return					An error code.		*/
    t_max_err db_query(t_database* db, t_db_result** dbresult, const char* sql, ...);


    /**	Execute a SQL query on the database.
         @ingroup	database
         @param		db			The #t_database pointer for your database instance.
         @param		dbresult	The address of a #t_db_result pointer.
                                If the pointer is passed-in set to NULL then a new dbresult will be created.
                                If the pointer is not NULL then it is assumed to be a valid dbresult, which will be filled in with the query results.
                                When you are done with the dbresult you should free it with object_free().
         @param		sql			A C-string containing a valid SQL query.
         @return				An error code.		*/
    t_max_err db_query_direct(t_database* db, t_db_result** dbresult, const char* sql);


    /**	Execute a SQL query on the database, temporarily overriding the database's error logging attribute.
        @ingroup	database
        @param		db			The #t_database pointer for your database instance.
        @param		dbresult	The address of a #t_db_result pointer.
                                If the pointer is passed-in set to NULL then a new dbresult will be created.
                                If the pointer is not NULL then it is assumed to be a valid dbresult, which will be filled in with the query results.
                                When you are done with the dbresult you should free it with object_free().
        @param		sql			A C-string containing a valid SQL query, possibly with sprintf() formatting codes.
        @param		...			If an sprintf() formatting codes are used in the sql string, these values will be interpolated into the sql string.
        @return					An error code.		*/
    t_max_err db_query_silent(t_database* db, t_db_result** dbresult, const char* sql, ...);


    /**	Determine the id (key) number for the most recent INSERT query executed on the database.
        @ingroup	database
        @param		db			The #t_database pointer for your database instance.
        @param		id			The address of a variable to hold the result on return.
        @return					An error code.		*/
    t_max_err db_query_getlastinsertid(t_database* db, long* id);


    /**	Create a new table in a database.
        @ingroup	database
        @param		db			The #t_database pointer for your database instance.
        @param		tablename	The name to use for the new table.
                                The new table will be created with one column, which holds the primary key for the table,
                                and is named according the form {tablename}_id.
        @return					An error code.		*/
    t_max_err db_query_table_new(t_database* db, const char* tablename);


    /**	Add a new column to an existing table in a database.
        @ingroup	database
        @param		db			The #t_database pointer for your database instance.
        @param		tablename	The name of the table to which the column should be added.
        @param		columnname	The name to use for the new column.
        @param		columntype	The SQL type for the data that will be stored in the column.
                                For example: "INTEGER" or "VARCHAR"
        @param		flags		If you wish to specify any additional information for the column, then pass that here.
                                Otherwise pass NULL.
        @return					An error code.		*/
    t_max_err db_query_table_addcolumn(t_database* db, const char* tablename, const char* columnname, const char* columntype, const char* flags);


    /**	Begin a database transaction.
        When you are working with a file-based database, then the database will not be flushed to disk until db_transacation_end() is called.
        This means that you can _much_ more efficiently execute a sequence of queries in one transaction rather than independently.

        That database object reference counts transactions, so it is possible nest calls to db_transacation_start() and db_transacation_end().
        It is important to balance all calls with db_transacation_end() or the database contents will never be flushed to disk.

        @ingroup	database
        @param		db			The #t_database pointer for your database instance.
        @return					An error code.		*/
    t_max_err db_transaction_start(t_database* db);


    /**	Finalize a database transaction.
        @ingroup	database
        @param		db			The #t_database pointer for your database instance.
        @return					An error code.		*/
    t_max_err db_transaction_end(t_database* db);


    /**	Force any open transactions to close.
        @ingroup	database
        @param		db			The #t_database pointer for your database instance.
        @return					An error code.		*/
    t_max_err db_transaction_flush(t_database* db);


    // DB VIEWS

    /**	A database view is a way of looking at a particular set of records in the database.
        This particular set of records is defined with a standard SQL query,
        and the view maintains a copy of the results of the query internally.
        Any time the database is modified the internal result set is updated,
        and any objects listening to the view are notified via object_notify().

        @ingroup	database
        @param		db			The #t_database pointer for your database instance.
        @param		sql			A SQL query that defines the set of results provided by the view.
        @param		dbview		The address of a NULL #t_db_view pointer which will be set with the new view upon return.
        @return					An error code.		*/
    t_max_err db_view_create(t_database* db, const char* sql, t_db_view** dbview);


    /**	Remove a database view created using db_view_create().
        @ingroup	database
        @param		db			The #t_database pointer for your database instance for which this view was created.
        @param		dbview		The address of the #t_db_view pointer for the view.
                                This pointer will be freed and set NULL upon return.
        @return					An error code.		*/
    t_max_err db_view_remove(t_database* db, t_db_view** dbview);


    /**	Fetch the pointer for a #t_db_view's query result.
        @ingroup	database
        @param		dbview		The #t_db_view pointer for your database view instance.
        @param		result		The address of a pointer to a #t_db_result object.
                                This pointer will be overwritten with the view's result pointer upon return.
        @return					An error code.		*/
    t_max_err db_view_getresult(t_db_view* dbview, t_db_result** result);


    /**	Set the query used by the view.
        @ingroup	database
        @param		dbview		The #t_db_view pointer for your database view instance.
        @param		newquery	The SQL string to define a new query for the view, replacing the old query.
        @return					An error code.		*/
    t_max_err db_view_setquery(t_db_view* dbview, char* newquery);


    // DB RESULTS

    /**	Return the next record from a set of results that you are walking.
        When you are returned a result from a query of the database,
        the result is prepared for walking the results from the beginning.
        You can also reset the result manually to the beginning of the record list
        by calling db_result_reset().

        @ingroup	database
        @param		result		The #t_db_result pointer for your query results.
        @return					An array of C-Strings with the values for every requested column (field) of a database record.
                                To find out how many columns are represented in the array, use db_result_numfields(). */
    char** db_result_nextrecord(t_db_result* result);


    /**	Reset the interface for walking a result's record list to the first record.
        @ingroup	database
        @param		result		The #t_db_result pointer for your query results.	 */
    void db_result_reset(t_db_result* result);


    /**	Zero-out a database result.
        @ingroup	database
        @param		result		The #t_db_result pointer for your query results.	*/
    void db_result_clear(t_db_result* result);


    /**	Return a count of all records in the query result.
        @ingroup	database
        @param		result		The #t_db_result pointer for your query results.
        @return					The count of records in the query result.	*/
    long db_result_numrecords(t_db_result* result);


    /**	Return a count of all fields (columns) in the query result.
        @ingroup	database
        @param		result		The #t_db_result pointer for your query results.
        @return					The count of fields in the query result.	*/
    long db_result_numfields(t_db_result* result);


    /**	Return the name of a field specified by its index number.
        @ingroup	database
        @param		result		The #t_db_result pointer for your query results.
        @param		fieldindex	The zero-based index number of the field (column) in the result.
        @return					A C-String with the name of the field.		*/
    char* db_result_fieldname(t_db_result* result, long fieldindex);


    /**	Return a single value from a result according to its index and field coordinates.
        @ingroup	database
        @param		result			The #t_db_result pointer for your query results.
        @param		recordindex		The zero-based index number of the record (row) in the result.
        @param		fieldindex		The zero-based index number of the field (column) in the result.
        @return						A C-String with the content of the specified cell in the result.	*/
    char* db_result_string(t_db_result* result, long recordindex, long fieldindex);

    /**	Return a single value from a result according to its index and field coordinates.
        @ingroup	database
        @param		result			The #t_db_result pointer for your query results.
        @param		recordindex		The zero-based index number of the record (row) in the result.
        @param		fieldindex		The zero-based index number of the field (column) in the result.
        @return						The content of the specified cell from the result scanned out to a long int.	*/
    long db_result_long(t_db_result* result, long recordindex, long fieldindex);

    /**	Return a single value from a result according to its index and field coordinates.
        @ingroup	database
        @param		result			The #t_db_result pointer for your query results.
        @param		recordindex		The zero-based index number of the record (row) in the result.
        @param		fieldindex		The zero-based index number of the field (column) in the result.
        @return						The content of the specified cell from the result scanned out to a float.		*/
    float db_result_float(t_db_result* result, long recordindex, long fieldindex);

    /**	Return a single value from a result according to its index and field coordinates.
        The value will be coerced from an expected datetime field into seconds.
        @ingroup	database
        @param		result			The #t_db_result pointer for your query results.
        @param		recordindex		The zero-based index number of the record (row) in the result.
        @param		fieldindex		The zero-based index number of the field (column) in the result.
        @return						The datetime represented in seconds.							*/
    t_ptr_uint db_result_datetimeinseconds(t_db_result* result, long recordindex, long fieldindex);


    // UTILITIES

    /**	A utility to convert from a sql datetime string into seconds.
        @ingroup	database
        @param		string			A C-string containing a date and time in SQL format.
        @param		date			The datetime represented in seconds upon return. */
    void db_util_stringtodate(const char* string, t_ptr_uint* date);


    /**	A utility to convert from seconds into a sql-ready datetime string.
        @ingroup	database
        @param		date			The datetime represented in seconds.
        @param		string			The address of a valid C-string
                                    whose contents will be set to a SQL-ready string format upon return. */
    void db_util_datetostring(const t_ptr_uint date, char* string);


    END_USING_C_LINKAGE

}} // namespace c74::max
