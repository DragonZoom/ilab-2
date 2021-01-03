#pragma once

#include <exception>
#include <iostream>
#include <string>

namespace matrix
{

	template< typename T >
	class IBuff_t
	{
	public:

		IBuff_t             (const IBuff_t&) = delete;
		IBuff_t& operator = (const IBuff_t&) = delete;

	protected:

		T* m_data = nullptr;
		size_t m_size = 0, m_used = 0;

	protected:

		IBuff_t() = default;
		
		explicit IBuff_t(size_t size)
			: m_data((size == 0) ? nullptr : static_cast<T*>(::operator new(sizeof(T) * size)))
			, m_size(size)
			, m_used(0)
		{}

		~IBuff_t()
		{
			for (size_t k = 0; k < m_used; ++k)
			{
				(m_data + k)->~T();
			}
			::operator delete[] (m_data);
		}

		IBuff_t(IBuff_t&& that) noexcept
			: IBuff_t()
		{
			swap(that);
		}

		IBuff_t& operator = (IBuff_t&& that) noexcept
		{
			IBuff_t tmp(std::move(that));
			swap(tmp);
		}

		
		size_t size() const { return m_size; }

		void swap(IBuff_t& that) noexcept
		{
			std::swap(m_data, that.m_data);
			std::swap(m_size, that.m_size);
			std::swap(m_used, that.m_used);
		}

		void clear()
		{
			*this = IBuff_t();
		}
	};


}//namespace matrix