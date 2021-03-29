/// @file
///	@ingroup 	minlib
///	@copyright	Copyright 2018 The Min-Lib Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#include <thread>

namespace c74::min::lib {

    
    ///	A generic circular buffer designed specifically for access from a single thread.
    ///
    ///	Allows items of a single datatype to be written into a vector that wraps around to the beginning whenever the end is exceeded.
    ///	Reading items from the vector can then return chunks of the N most recent items.
    ///	The initial design of this class is for use as an audio sample delay buffer,
    ///	however it's functionality has been generalized to allow for other datatypes and applications.

    template<class T>
    class circular_storage {
    public:
        /// Constructor specifies a fixed amount of storage for the container.
        ///	If you want a different amount of storage, create a new container and dispose of the one you don't want.
        /// @param	item_count	The number of items to be stored in the container.

        explicit circular_storage(std::size_t item_count)
        : m_items(item_count)
        , m_size(item_count) {}


        circular_storage(std::pair<size_t, size_t> capacity_and_size)
        : m_items(capacity_and_size.first)
        , m_size(capacity_and_size.second) {}


        /// Write a block of items into the container.
        ///	@param	new_input	A block of items to add. May not be more items than the size (item_count) of the container.

        void write(const std::vector<T>& new_input) {
            assert(check_thread());
            assert(new_input.size() <= size());

            std::size_t count         = new_input.size();
            std::size_t roomRemaining = size() - m_index;
            bool        wrap          = false;

            if (count > roomRemaining) {
                count = roomRemaining;
                wrap  = true;
            }

            std::copy_n(new_input.begin(), count, m_items.begin() + m_index);
            m_index += count;

            if (wrap) {
                std::size_t offset = count;

                count = new_input.size() - offset;
                std::copy_n(new_input.begin() + offset, count, m_items.begin());
                m_index = count;
            }
        }


        /// Write a single item into the container.
        ///	@param	new_input	An item to add.

        void write(const T& new_input) {
            assert(check_thread());

            m_items[m_index] = new_input;
            ++m_index;
            if (m_index >= size())
                m_index = 0;

            m_items[m_index] = T();    // delays <1 sample require us to clear the old value at new index
        }


        /// Read a block of items out from the container. This is the same as calling the head() method.
        ///	These will be the N most recent items added to the history.
        ///	@param	output	A place to write the block of items from the buffer.
        ///					The size of this buffer will determine the number of items to request.
        ///					May not be larger than the size of the container.
        /// @see	head()

        void read(std::vector<T>& output) {
            head(output);
        }


        /// Read a block of items out from the container.
        ///	These will be the N most recent items added to the history.
        ///	@param	output	A place to write the block of things from the buffer.
        ///					The size of this buffer will determine the number of items to request.
        ///					May not be larger than the size of the container.
        ///	@see	tail()

        void head(std::vector<T>& output) {
            assert(check_thread());
            assert(size() >= output.size());

            auto count = static_cast<long>(output.size());
            auto start = static_cast<long>(m_index - count);
            bool wrap  = false;

            while (start < 0) {
                count = -start;
                start = static_cast<long>(size()) + start;
                wrap  = true;
            }

            std::copy_n(m_items.begin() + start, count, output.begin());

            if (wrap) {
                std::size_t offset = count;

                count = static_cast<long>(output.size() - offset);
                std::copy_n(m_items.begin(), count, output.begin() + offset);
            }
        }


        /// Read a block of items out from the container.
        ///	These will be the N oldest items added in the history.
        ///	@param	output	A place to write the block of items from the buffer.
        ///					The size of this buffer will determine the number of items to request.
        ///					May not be larger than the size of the container.
        ///	@see	head()

        void tail(std::vector<T>& output) {
            assert(check_thread());
            assert(size() >= output.size());

            auto count = static_cast<long>(output.size());
            auto start = static_cast<long>(m_index % size());
            bool wrap  = false;

            while (start + count > size()) {
                count = static_cast<long>(size()) - start;
                wrap  = true;
            }

            std::copy_n(m_items.begin() + start, count, output.begin());

            if (wrap) {
                std::size_t offset = count;

                count = static_cast<long>(output.size() - offset);
                std::copy_n(m_items.begin(), count, output.begin() + offset);
            }
        }


        /// Read a single item out from the container.
        ///	This will be the oldest item in the history.
        /// @param	offset	An optional parameter for getting an item that is N items newer than the oldest value.
        ///	@return	output	The item from the buffer.
        ///	@see	head()

        T tail(size_t offset = 0) {
            // TODO: benchmark / evaluate use of modulo here
            return m_items[(m_index + offset) % size()];
        }


        ///	Zero the contents without resizing.

        void zero() {
            assert(check_thread());
            T zero_value = T();    // call the constructor to ensure zero initialization
            std::fill(m_items.begin(), m_items.end(), zero_value);
        }


        /// Clear the contents, leaving internal vector with a size of zero.

        void clear() {
            assert(check_thread());
            m_items.clear();
            m_size = 0;
        }


        ///	Return the item count of the container.
        /// @return	The item count of the container.

        std::size_t size() {
            return m_size;
        }


        ///	Return the capacity (or maximum size) of the container.
        /// @return	The capacity (or maximum size) of the container.

        std::size_t capacity() {
            return m_items.size();
        }

        ///	Change the number of items in the container.
        /// Note that this does not change the capacity of the container,
        /// which can only be set when the container is created.
        /// @param	new_size	The new item count for the circular storage container.

        void resize(std::size_t new_size) {
            assert(new_size <= m_items.size());
            m_size = new_size;
            // if m_index is out of bounds after resize, we zero the memory to prevent invalid output from history
            if (m_index >= m_size) {
                m_index = 0;
                zero();
            }
        }


        /// Get a reference to an individual item by index from the container.
        /// @param	index	The index of the item to fetch.
        /// @return			A reference to the item.

        T& item(std::size_t index) {
            assert(index <= m_items.size());
            return m_items[index];
        }

    private:
        /// Confirm that calls to multiple public methods are happening on the same thread.
        /// First call to this method will set the m_thread variable.
        /// Every call after will compare to the current thread id to m_thread.
        /// @return         false if the thread id changes between calls, else true

        bool check_thread() {
            if (m_thread == null_thread) {
                m_thread = {std::this_thread::get_id()};
                return true;
            }
            else {
                return std::this_thread::get_id() == m_thread;
            }
        }

        std::vector<T>  m_items;        ///< storage for the circular buffer's data
        std::size_t     m_index{};      ///< location of the record head
        std::size_t     m_size;         ///< the size of the circular buffer (may be different from the amount of allocated storage)
        std::thread::id m_thread;       ///< used to ensure we don't access unsafely from multiple threads
        std::thread::id null_thread;    ///< save the default constructor output to catch the first call in check_thread()
    };


}    // namespace c74::min::lib
