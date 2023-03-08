/*
Copyright	(c)	2016	Michael	Welter

This	file	is	part	of	numpycpp.

numpycpp	is	free	software:	you	can	redistribute	it	and/or	modify
it	under	the	terms	of	the	GNU	General	Public	License	as	published
the	Free	Software	Foundation,	either	version	3	of	the	License,	or
(at	your	option)	any	later	version.

numpycpp	is	distributed	in	the	hope	that	it	will	be	useful,
but	WITHOUT	ANY	WARRANTY;	without	even	the	implied	warranty	of
MERCHANTABILITY	or	FITNESS	FOR	A	PARTICULAR	PURPOSE.	See	the
GNU	General	Public	License	for	more	details.

You	should	have	received	a	copy	of	the	GNU	General	Public	License
along	with	numpycpp.	If	not,	see	<http://www.gnu.org/licenses/>.
*/
#ifndef	__CXX_NUMPY_H__
#define	__CXX_NUMPY_H__
#define	NPY_NO_DEPRECATED_API	NPY_1_7_API_VERSION

#include	<boost/python.hpp>
#include	<boost/python/object.hpp>
#include	assert.h

//

namespace	boost	{	namespace	python	{

namespace	numpy	{

namespace	mw_py_impl
{
//	see	here	http://stackoverflow.com/questions/16454987/c-generate-a-compile-error-if-user-tries-to-instantiate-a-class-template-wiho
template<class	T>
class	incomplete;
};


enum	{
MAX_DIM	=	32
};


typedef	Py_ssize_t	ssize_t;


void	importNumpyAndRegisterTypes();


template<class	T>
int	getItemtype()
{
//	see	here	http://stackoverflow.com/questions/16454987/c-generate-a-compile-error-if-user-tries-to-instantiate-a-class-template-wiho
enum	{	S	=	sizeof(mw_py_impl::incomplete<T>)	};	//	will	generate	an	error	for
return	-1;
}

int	getItemtype(const	object	&arr);


#define	NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(T)\
template<>	int	getItemtype<T>();

NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(PyObject*)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(float)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(double)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(int)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(long)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(long	long)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(short)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(char)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(bool)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(unsigned	int)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(unsigned	long)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(unsigned	short)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(unsigned	char)
NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE(unsigned	long	long)
#undef	NUMPY_HPP_FWD_DECLARE_GET_ITEMTYPE

object	zeros(int	rank,	const	Py_ssize_t	*dims,	int	type);

object	empty(int	rank,	const	Py_ssize_t	*dims,	int	type);

template<class	T>
bool	isCompatibleType(int	id);

class	arraytbase
{
protected:
object	obj;	//	for	reference	counting
PyObject*	objptr;

arraytbase(object	const	&a,	int	typesize);
void	construct(object	const	&a,	int	typesize);

public:
/*
*/
arraytbase(object	const	&a	=	object());
typedef	Py_ssize_t	ssize_t;

const	ssize_t*	shape()	const;
const	ssize_t*	dims()	const	{	return	shape();	}
const	ssize_t*	strides()	const;
int	itemtype()	const;
int	itemsize()	const;
int	rank()	const;
int	ndim()	const	{	return	rank();	}
bool	isWriteable()	const;
bool	isCContiguous()	const;
bool	isFContiguous()	const;
const	object&	getObject()	const	{	return	obj;	}

#define	NUMPY_HPP_OFFSET1	\
x*m[0]
#define	NUMPY_HPP_OFFSET2	\
NUMPY_HPP_OFFSET1	+	y*m[1]
#define	NUMPY_HPP_OFFSET3	\
NUMPY_HPP_OFFSET2	+	z*m[2]
#define	NUMPY_HPP_OFFSET4	\
NUMPY_HPP_OFFSET3	+	w*m[3]


inline	ssize_t	offset(int	x)	const
{
const	Py_ssize_t*	m	=	strides();
assert((unsigned	int)x<shape()[0]);
return	NUMPY_HPP_OFFSET1;
}

inline	ssize_t	offset(int	x,	int	y)	const
{
const	Py_ssize_t*	m	=	strides();
assert((Py_ssize_t)x<shape()[0]);
assert((Py_ssize_t)y<shape()[1]);
return	NUMPY_HPP_OFFSET2;
}

inline	ssize_t	offset(int	x,	int	y,	int	z)	const
{
const	Py_ssize_t*	m	=	strides();
assert((Py_ssize_t)x<shape()[0]);
assert((Py_ssize_t)y<shape()[1]);
assert((Py_ssize_t)z<shape()[2]);
return	NUMPY_HPP_OFFSET3;
}

inline	ssize_t	offset(int	x,int	y,	int	z,	int	w)	const
{
const	Py_ssize_t*	m	=	strides();
assert((Py_ssize_t)x<shape()[0]);
assert((Py_ssize_t)y<shape()[1]);
assert((Py_ssize_t)z<shape()[2]);
assert((Py_ssize_t)w<shape()[3]);
return	NUMPY_HPP_OFFSET4;
}

inline	ssize_t	offset(const	int	*c)	const
{
const	Py_ssize_t*	m	=	strides();
int	r	=	rank();
Py_ssize_t	q	=	0;
for(int	i=0;	i<r;	++i)
{
assert((Py_ssize_t)(c[i])<shape()[i]);
q	+=	c[i]*m[i];
}
return	q;
}

#undef	NUMPY_HPP_OFFSET1
#undef	NUMPY_HPP_OFFSET2
#undef	NUMPY_HPP_OFFSET3
#undef	NUMPY_HPP_OFFSET4

char*	bytes();
const	char*	bytes()	const	{	return	const_cast<arraytbase*>(this)->bytes();	}
};

template<class	T>
class	arrayt	:	public	arraytbase
{
public:
arrayt()	:	arraytbase()	{}
arrayt(arraytbase	const	&a)	:	arraytbase(a.getObject(),	sizeof(T))
{
}

arrayt(object	const	&a)	:	arraytbase(a,	sizeof(T))
{
}

void	init(object	const	&a_)
{
this->~arrayt<T>();
new	(this)	arrayt<T>(a_);
}

T&	operator()(int	x)
{
return	*((T*)(arraytbase::bytes()+offset(x)));
}

T&	operator()(int	x,	int	y)
{
return	*((T*)(arraytbase::bytes()+offset(x,y)));
}

T&	operator()(int	x,	int	y,	int	z)
{
return	*((T*)(arraytbase::bytes()+offset(x,y,z)));
}

T&	operator()(int	x,	int	y,	int	z,	int	w)
{
return	*((T*)(arraytbase::bytes()+offset(x,y,z,w)));
}

T&	operator()(int	*c)
{
return	*((T*)(arraytbase::bytes()+offset(c)));
}

T&	operator[](int	i)
{
return	*((T*)(arraytbase::bytes()+offset(i)));
}

T*	data()	{	return	reinterpret_cast<T*>(bytes());	}


T	operator()(int	x)	const
{
return	*((T*)(arraytbase::bytes()+offset(x)));
}

T	operator()(int	x,	int	y)	const
{
return	*((T*)(arraytbase::bytes()+offset(x,y)));
}

T	operator()(int	x,	int	y,	int	z)	const
{
return	*((T*)(arraytbase::bytes()+offset(x,y,z)));
}

T	operator()(int	x,	int	y,	int	z,	int	w)	const
{
return	*((T*)(arraytbase::bytes()+offset(x,y,z,w)));
}

T	operator()(int	*c)	const
{
return	*((T*)(arraytbase::bytes()+offset(c)));
}

T	operator[](int	i)	const
{
return	*((T*)(arraytbase::bytes()+offset(i)));
}

const	T*	data()	const	{	return	reinterpret_cast<T*>(bytes());	}
};


namespace	mw_py_impl
{
template<class	T,	class	Idx1,	class	Idx2,	class	Idx3>
inline	void	gridded_data_ccons(T*	dst,	const	T*	src,
const	Idx1	*dims,
const	Idx2	*dst_strides,
const	Idx3	*src_strides,
boost::mpl::int_<0>)
{
for	(int	i=0;	i<dims[0];	++i)
{
new	(dst)	T(*src);
dst	+=	dst_strides[0];
src	+=	src_strides[0];
}
}

template<class	T,	class	Idx1,	class	Idx2,	class	Idx3,	int	dim>
inline	void	gridded_data_ccons(T*	dst,	const	T*	src,
const	Idx1*	dims,
const	Idx2	*dst_strides,
const	Idx3	*src_strides,
boost::mpl::int_<dim>)
{
for	(int	i=0;	i<dims[dim];	++i)
{
gridded_data_ccons(dst,	src,	dims,	dst_strides,	src_strides,	boost::mpl::int_<dim-1>());
dst	+=	dst_strides[dim];
src	+=	src_strides[dim];
}
}
}


template<class	T,	int	rank>
static	object	copy(const	int*	dims,	const	T*	src,	const	int	*strides)
{
int	item_type	=	getItemtype<T>();
typename	arrayt<T>::ssize_t	arr_dims[MAX_DIM];
for	(int	i=0;	i<rank;	++i)	arr_dims[i]	=	dims[i];
arrayt<T>	arr(empty(rank,	arr_dims,	item_type));

typename	arrayt<T>::ssize_t	arr_strides[MAX_DIM];
for	(int	i=0;	i<rank;	++i)	arr_strides[i]	=	arr.strides()[i]/arr.itemsize();

T*	dst	=	arr.data();
mw_py_impl::gridded_data_ccons<T>(dst,	src,	dims,	arr_strides,	strides,	boost::mpl::int_<rank-1>());

return	arr.getObject();
}

template<class	T,	int	rank>
static	void	copy(T*	dst,	const	int	*dims,	const	int*	strides,	const	object
{
arrayt<T>	arr(pyarr);

typename	arrayt<T>::ssize_t	arr_strides[MAX_DIM];
for	(int	i=0;	i<rank;	++i)	{
assert(arr.shape()[i]	==	dims[i]);
arr_strides[i]	=	arr.strides()[i]/arr.itemsize();
}

const	T*	src	=	arr.data();
mw_py_impl::gridded_data_ccons<T>(dst,	src,	arr.shape(),	strides,	arr_strides,	boost::mpl::int_<rank-1>());
}


}	}	}

#endif											
