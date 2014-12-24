#ifndef MSHADOW_EXPRESSION_H_
#define MSHADOW_EXPRESSION_H_
/*!
 * \file expression.h
 * \brief definitions of abstract expressions and expressions template
 * \author Tianqi Chen, Bing Xu
 */
#include "./base.h"

namespace mshadow {
/*!
 * \brief namespace for abstract expressions and expressions template,
 *        have no dependecy on tensor.h,
 *        These data structure takes no charge in computations,
 *        they are only used to define operations and represent expression in a symbolic way
 */
namespace expr {
/*! \brief type of expressions */
namespace type {
// type expression type are defined as bitmask
// subtype relationshop kRValue < kMapper < kPull < kComplex
/*! \brief this expression directly correspnds to a data class, can be used to assign data */
const int kRValue = 0;
/*! \brief expression contains element-wise tensor operations, map a expression to same shape */
const int kMapper = 1;
/*!
 * \brief expression that can be chained with other expressiones
 *    Usually it have function Eval(i,j) defined, which pulls the result (i, j) from input
 *    expression and output the result at certain position.
 */
const int kChainer = 3;
/*! \brief othercase: e.g dot product */
const int kComplex = 7;
}  // namespace type
/*!
 * \brief expression engine that actually interprets these expressions
 *   this is a function template that needed to be implemented for specific expressions
 * \tparam Saver the save method
 * \tparam RValue the type of RValue to be saved
 * \sa namespace sv
 */
template<typename Saver, typename RValue>
struct ExpEngine {
  /*! \brief defines how expression exp can be evaluated and stored into dst */
  template<typename EType>
  inline static void Eval(RValue& dst, const EType &exp);
};
/*!
 * \brief base class for expression
 * \tparam SubType inheritated class must put their type into this parameter
 * \tparam exp_type expression type, see namespace type
 */
template<typename SubType, int exp_type>
struct Exp {
 public:
  /*! \return  subtype instance of current class */
  inline const SubType& self(void) const {
    return *static_cast<const SubType*>(this);
  }
  /*! \return reference of subtype instance of current class */
  inline SubType& refself(void) {
    return *static_cast<SubType*>(this);
  }
};
/*!
 * \brief scalar expression 
 * \tparam DType the data type of the scalar
 */
template<typename DType>
struct ScalarExp: public Exp<ScalarExp<DType>, type::kMapper> {
  /*! \brief scalar value */
  DType scalar_;
  /*! \brief constructor, must be implicit for implicit conversion */
  ScalarExp(DType scalar) : scalar_(scalar) {}
};
/*! \brief create an scalar expression */
template<typename DType>
inline ScalarExp<DType> scalar(DType s) {
  return ScalarExp<DType>(s);
}
/*!
 * \brief typecast expression, cast the type of elements
 * \tparam DType the target type we want to cast into
 * \tparam EType the type of the source expression
 * \tparam etype the type of expression after cast
 */
template<typename DType, typename EType, int etype>
struct TypecastExp: public Exp<TypecastExp<DType, EType, etype>, etype> {
  const EType &exp;
  /*! \brief constructor */
  explicit TypecastExp(const EType &e) : exp(e) {}  
};
/*! \brief create an scalar expression */
template<typename DType, typename EType, int etype>
inline TypecastExp<DType, EType, (etype|type::kMapper)> tcast(const Exp<EType, etype> &exp) {
  return TypecastExp<DType, EType, (etype|type::kMapper)>(exp.self());
}
/*! \brief represent a transpose expression of a container */
template<typename EType>
struct TransposeExp: public Exp<TransposeExp<EType>, type::kChainer> {
  /*! \brief expression to be transposed */
  const EType &exp;
  /*! \brief constructor */
  explicit TransposeExp(const EType &e) : exp(e) {}
  /*! \brief transpose expression */
  inline const EType &T(void) const {
    return exp;
  }
};
/*!
 * \brief base class of all rvalues
 * \tparam Container the actually class of data container, e.g. Tensor1D
 * \tparam DataType the element data type of each element in the container
 */
template<typename Container, typename DType>
class RValueExp: public Exp<Container, type::kRValue> {
 public:
  /*!
   *\brief transpose of a matrix
   *\return transpose of current expression
   */
  inline const TransposeExp<Container> T(void) const {
    return TransposeExp<Container>(this->self());
  }
  /*! \brief operator overload */
  inline Container &operator+=(default_real_t s) {
    ExpEngine<sv::plusto, Container>::Eval(this->refself(), scalar(s));
    return this->refself();
  }
  /*! \brief operator overload */ 
  inline Container &operator-=(default_real_t s) {
    ExpEngine<sv::minusto, Container>::Eval(this->refself(), scalar(s));
    return this->refself();
  }
  /*! \brief operator overload */
  inline Container &operator*=(default_real_t s) {
    ExpEngine<sv::multo, Container>::Eval(this->refself(), scalar(s));
    return this->refself();
  }
  /*! \brief operator overload */
  inline Container &operator/=(default_real_t s) {
    ExpEngine<sv::divto, Container>::Eval(this->refself(), scalar(s));
    return this->refself();
  }
  /*! \brief operator overload */
  inline Container &__assign(default_real_t s) {
    ExpEngine<sv::saveto, Container>::Eval(this->refself(), scalar(s));
    return this->refself();
  }
  /*! \brief implementation of operator=, note that we can not define container = container */
  template<typename E>
  inline Container &__assign(const Exp<E, type::kMapper> &exp) {
    ExpEngine<sv::saveto, Container>::Eval(this->refself(), exp.self());
    return this->refself();
  }
  /*! \brief implementation of operator=, note that we can not define conatiner = container */
  template<typename E>
  inline Container &__assign(const Exp<E, type::kChainer> &exp) {
    ExpEngine<sv::saveto, Container>::Eval(this->refself(), exp.self());
    return this->refself();
  }
  /*! \brief implementation of operator=, note that we can not define container = container */
  template<typename E>
  inline Container &__assign(const Exp<E, type::kComplex> &exp) {
    ExpEngine<sv::saveto, Container>::Eval(this->refself(), exp.self());
    return this->refself();
  }
  /*! \brief implementation of operator+= */
  template<typename E, int etype>
  inline Container &operator+=(const Exp<E, etype> &exp) {
    ExpEngine<sv::plusto, Container>::Eval(this->refself(), exp.self());
    return this->refself();
  }
  /*! \brief implementation of operator-= */
  template<typename E, int etype>
  inline Container &operator-=(const Exp<E, etype> &exp) {
    ExpEngine<sv::minusto, Container>::Eval(this->refself(), exp.self());
    return this->refself();
  }
  /*! \brief implementation of operator*= */
  template<typename E, int etype>
  inline Container &operator*=(const Exp<E, etype> &exp) {
    ExpEngine<sv::multo, Container>::Eval(this->refself(), exp.self());
    return this->refself();
  }
  /*! \brief implementation of operator/= */
  template<typename E, int etype>
  inline Container &operator/=(const Exp<E, etype> &exp) {
    ExpEngine<sv::divto, Container>::Eval(this->refself(), exp.self());
    return this->refself();
  }
};
/*!
 * \brief matrix multiplication expression dot(lhs[.T], rhs[.T])
 * \tparam TA type of lhs
 * \tparam TB type of rhs
 * \tparam ltrans whether lhs is transposed
 * \tparam rtrans whether rhs is transposed
 */
template<typename TA, typename TB, bool ltrans, bool rtrans>
struct DotExp: public Exp<DotExp<TA, TB, ltrans, rtrans>, type::kComplex> {
  /*! \brief left operand */
  const TA &lhs_;
  /*! \brief right operand */
  const TB &rhs_;
  /*! \brief scale over result */
  default_real_t scale_;
  /*! \brief constructor */
  explicit DotExp(const TA &lhs, const TB &rhs, default_real_t scale)
      : lhs_(lhs), rhs_(rhs), scale_(scale) {}
};
// definition of dot expression
/*! \brief dot operator def */
template<typename TA, typename TB, typename DType>
inline DotExp<TA, TB, false, false>
dot(const RValueExp<TA, DType> &lhs, const RValueExp<TB, DType> &rhs) {
  return DotExp<TA, TB, false, false>(lhs.self(), rhs.self(), 1.0f);
}
/*! \brief dot operator def */
template<typename TA, typename TB, typename DType>
inline DotExp<TA, TB, true, false>
dot(const TransposeExp<TA> &lhs, const RValueExp<TB, DType> &rhs) {
  return DotExp<TA, TB, true, false>(lhs.exp, rhs.self(), 1.0f);
}
/*! \brief dot operator def */
template<typename TA, typename TB, typename DType>
inline DotExp<TA, TB, false, true>
dot(const RValueExp<TA, DType> &lhs, const TransposeExp<TB> &rhs) {
  return DotExp<TA, TB, false, true>(lhs.self(), rhs.exp, 1.0f);
}
/*! \brief dot operator def */
template<typename TA, typename TB>
inline DotExp<TA, TB, true, true>
dot(const TransposeExp<TA> &lhs, const TransposeExp<TB> &rhs) {
  return DotExp<TA, TB, true, true>(lhs.exp, rhs.exp, 1.0f);
}
/*! \brief dot operator def */
template<typename TA, typename TB, bool ltrans, bool rtrans>
inline DotExp<TA, TB, ltrans, rtrans>
operator*(const DotExp<TA, TB, ltrans, rtrans> &lhs, default_real_t rhs) {
  return DotExp<TA, TB, ltrans, rtrans>(lhs.lhs_, lhs.rhs_, lhs.scale_ * rhs);
}
/*! \brief scale of dot operation */
template<typename TA, typename TB, bool ltrans, bool rtrans>
inline DotExp<TA, TB, ltrans, rtrans>
operator*(default_real_t lhs, const DotExp<TA, TB, ltrans, rtrans> &rhs) {
  return DotExp<TA, TB, ltrans, rtrans>(rhs.lhs_, rhs.rhs_, rhs.scale_ * lhs);
}
//---------------
// BinaryMapExp
// --------------
/*!
 * \brief binary map expression lhs [op] rhs
 * \tparam OP operator
 * \tparam TA type of lhs
 * \tparam TB type of rhs
 * \tparam etype expression type, sa namespace::type
 */
template<typename OP, typename TA, typename TB, int etype>
struct BinaryMapExp: public Exp< BinaryMapExp<OP, TA, TB, etype>, etype> {
  /*! \brief left operand */
  const TA &lhs_;
  /*! \brief right operand */
  const TB &rhs_;
  /*! \brief constructor */
  explicit BinaryMapExp(const TA &lhs, const TB &rhs)
      :lhs_(lhs), rhs_(rhs) {}
};

/*! \brief make expression */
template<typename OP, typename TA, typename TB, int ta, int tb>
inline BinaryMapExp<OP, TA, TB, (ta|tb|type::kMapper)>
MakeExp(const Exp<TA, ta> &lhs, const Exp<TB, tb> &rhs) {
  return BinaryMapExp<OP, TA, TB, (ta|tb|type::kMapper)>(lhs.self(), rhs.self());
}
/*! 
 * \brief short hand for MakeExp, usage F<op>(lhs, rhs). create a binary operation expression 
 * \param lhs left operand
 * \param rhs right operand
 * \tparam binary operator 
 * \tparam TA lhs expression
 * \tparam ta lhs expression type
 * \tparam TB rhs expression
 * \tparam tb rhs expression type
 * \sa mshadow::op
 */
template<typename OP, typename TA, typename TB, int ta, int tb>
inline BinaryMapExp<OP, TA, TB, (ta|tb|type::kMapper)>
F(const Exp<TA, ta> &lhs, const Exp<TB, tb> &rhs) {
  return MakeExp<OP>(lhs, rhs);
}
/*! \brief operator overload for const */
template<typename OP, typename TA, int ta>
inline BinaryMapExp<OP, TA, ScalarExp<default_real_t>, (ta|type::kMapper)>
F(const Exp<TA, ta> &lhs, const ScalarExp<default_real_t> &rhs) {
  return MakeExp<OP>(lhs, rhs);
}
/*! \brief operator overload for const */
template<typename OP, typename TB, int tb>
inline BinaryMapExp<OP, ScalarExp<default_real_t>, TB, (tb|type::kMapper)>
F(const ScalarExp<default_real_t> &lhs, const Exp<TB, tb> &rhs) {
  return MakeExp<OP>(lhs, rhs);
}
// operator rules
/*! \brief operator overload */
template<typename TA, typename TB, int ta, int tb>
inline BinaryMapExp<op::plus, TA, TB, (ta|tb|type::kMapper)>
operator+(const Exp<TA, ta> &lhs, const Exp<TB, tb> &rhs) {
  return MakeExp<op::plus>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TA, typename TB, int ta, int tb>
inline BinaryMapExp<op::minus, TA, TB, (ta|tb|type::kMapper)>
operator-(const Exp<TA, ta> &lhs, const Exp<TB, tb> &rhs) {
  return MakeExp<op::minus>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TA, typename TB, int ta, int tb>
inline BinaryMapExp<op::mul, TA, TB, (ta|tb|type::kMapper)>
operator*(const Exp<TA, ta> &lhs, const Exp<TB, tb> &rhs) {
  return MakeExp<op::mul>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TA, typename TB, int ta, int tb>
inline BinaryMapExp<op::div, TA, TB, (ta|tb|type::kMapper)>
operator/(const Exp<TA, ta> &lhs, const Exp<TB, tb> &rhs) {
  return MakeExp<op::div>(lhs, rhs);
}
// constant operators
/*! \brief operator overload */
template<typename TA, int ta>
inline BinaryMapExp<op::plus, TA, ScalarExp<default_real_t>, (ta|type::kMapper)>
operator+(const Exp<TA, ta> &lhs,  const ScalarExp<default_real_t> &rhs) {
  return MakeExp<op::plus>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TA, int ta>
inline BinaryMapExp<op::minus, TA, ScalarExp<default_real_t>, (ta|type::kMapper)>
operator-(const Exp<TA, ta> &lhs,  const ScalarExp<default_real_t> &rhs) {
  return MakeExp<op::minus>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TA, int ta>
inline BinaryMapExp<op::mul, TA, ScalarExp<default_real_t>, (ta|type::kMapper)>
operator*(const Exp<TA, ta> &lhs,  const ScalarExp<default_real_t> &rhs) {
  return MakeExp<op::mul>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TA, int ta>
inline BinaryMapExp<op::div, TA, ScalarExp<default_real_t>, (ta|type::kMapper)>
operator/(const Exp<TA, ta> &lhs,  const ScalarExp<default_real_t> &rhs) {
  return MakeExp<op::div>(lhs, rhs);
}
// constant operators 2
/*! \brief operator overload */
template<typename TB, int tb>
inline BinaryMapExp<op::plus, ScalarExp<default_real_t>, TB, (tb|type::kMapper)>
operator+(const ScalarExp<default_real_t> &lhs, const Exp<TB, tb> &rhs) {
  return MakeExp<op::plus>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TB, int tb>
inline BinaryMapExp<op::minus, ScalarExp<default_real_t>, TB, (tb|type::kMapper)>
operator-(const ScalarExp<default_real_t> &lhs, const Exp<TB, tb> &rhs) {
  return MakeExp<op::minus>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TB, int tb>
inline BinaryMapExp<op::mul, ScalarExp<default_real_t>, TB, (tb|type::kMapper)>
operator*(const ScalarExp<default_real_t> &lhs, const Exp<TB, tb> &rhs) {
  return MakeExp<op::mul>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TB, int tb>
inline BinaryMapExp<op::div, ScalarExp<default_real_t>, TB, (tb|type::kMapper)>
operator/(const ScalarExp<default_real_t> &lhs, const Exp<TB, tb> &rhs) {
  return MakeExp<op::div>(lhs, rhs);
}
//---------------
// UnaryMapExp
// --------------
/*!
 * \brief unary map expression op(src)
 * \tparam OP operator
 * \tparam TA type of src
 * \tparam etype expression type, sa namespace::type
 */
template<typename OP, typename TA, int etype>
struct UnaryMapExp: public Exp< UnaryMapExp<OP, TA, etype>, etype> {
  /*! \brief source expression */
  const TA &src_;
  /*! \brief constructor */
  explicit UnaryMapExp(const TA &src) : src_(src) {}
};

/*! \brief make expression */
template<typename OP, typename TA, int ta>
inline UnaryMapExp<OP, TA, (ta|type::kMapper)> MakeExp(const Exp<TA, ta> &src) {
  return UnaryMapExp<OP, TA, (ta|type::kMapper)>(src.self());
}
/*! 
 * \brief short hand for MakeExp, usage F<op>(src), create a unary operation expression 
 * \param src source expression
 * \tparam operator 
 * \tparam TA source expression
 * \tparam ta source expression type
 * \sa mshadow::op
 */
template<typename OP, typename TA, int ta>
inline UnaryMapExp<OP, TA, (ta|type::kMapper)> F(const Exp<TA, ta> &src) {
  return MakeExp<OP>(src);
}


}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXPRESSION_H_
