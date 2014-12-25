/*!
 *  Copyright (c) 2014 by Contributors
 * \file expression-inl.h
 * \brief definitions of operators in expression with respect to scalar
 *  this file will be included several times, each time with MACRO MSHADOW_SCALAR_ to be different types
 *
 * DO NOT add pragma once or macro guard
 * \author Tianqi Chen, Bing Xu
 */
namespace mshadow {
namespace expr {
// DotExp
/*! \brief dot operator def */
template<typename TA, typename TB, bool ltrans, bool rtrans>
inline DotExp<TA, TB, ltrans, rtrans, MSHADOW_SCALAR_>
operator*(const DotExp<TA, TB, ltrans, rtrans, MSHADOW_SCALAR_> &lhs,
          MSHADOW_SCALAR_ rhs) {
  return DotExp<TA, TB, ltrans, rtrans,
                MSHADOW_SCALAR_>(lhs.lhs_, lhs.rhs_, lhs.scale_ * rhs);
}
/*! \brief scale of dot operation */
template<typename TA, typename TB, bool ltrans, bool rtrans>
inline DotExp<TA, TB, ltrans, rtrans, MSHADOW_SCALAR_>
operator*(MSHADOW_SCALAR_ lhs,
          const DotExp<TA, TB, ltrans, rtrans, MSHADOW_SCALAR_> &rhs) {
  return DotExp<TA, TB, ltrans, rtrans,
                MSHADOW_SCALAR_>(rhs.lhs_, rhs.rhs_, rhs.scale_ * lhs);
}

/*! \brief operator overload for const */
template<typename OP, typename TA, int ta>
inline BinaryMapExp<OP, TA, ScalarExp<MSHADOW_SCALAR_>,
                    MSHADOW_SCALAR_, (ta|type::kMapper)>
F(const Exp<TA, MSHADOW_SCALAR_, ta> &lhs, MSHADOW_SCALAR_ rhs) {
  return MakeExp<OP>(lhs, scalar<MSHADOW_SCALAR_>(rhs));
}
/*! \brief operator overload for const */
template<typename OP, typename TB, int tb>
inline BinaryMapExp<OP, ScalarExp<MSHADOW_SCALAR_>, TB,
                    MSHADOW_SCALAR_, (tb|type::kMapper)>
F(MSHADOW_SCALAR_ lhs, const Exp<TB, MSHADOW_SCALAR_, tb> &rhs) {
  return MakeExp<OP>(scalar<MSHADOW_SCALAR_>(lhs), rhs);
}
// constant operators
/*! \brief operator overload */
template<typename TA, int ta>
inline BinaryMapExp<op::plus, TA, ScalarExp<MSHADOW_SCALAR_>,
                    MSHADOW_SCALAR_, (ta|type::kMapper)>
operator+(const Exp<TA, MSHADOW_SCALAR_, ta> &lhs,
          const MSHADOW_SCALAR_ &rhs) {
  return MakeExp<op::plus>(lhs, scalar<MSHADOW_SCALAR_>(rhs));
}
/*! \brief operator overload */
template<typename TA, int ta>
inline BinaryMapExp<op::minus, TA, ScalarExp<MSHADOW_SCALAR_>,
                    MSHADOW_SCALAR_, (ta|type::kMapper)>
operator-(const Exp<TA, MSHADOW_SCALAR_, ta> &lhs,
          const MSHADOW_SCALAR_ &rhs) {
  return MakeExp<op::minus>(lhs, scalar<MSHADOW_SCALAR_>(rhs));
}
/*! \brief operator overload */
template<typename TA, int ta>
inline BinaryMapExp<op::mul, TA, ScalarExp<MSHADOW_SCALAR_>,
                    MSHADOW_SCALAR_, (ta|type::kMapper)>
operator*(const Exp<TA, MSHADOW_SCALAR_, ta> &lhs,
          const ScalarExp<MSHADOW_SCALAR_> &rhs) {
  return MakeExp<op::mul>(lhs, rhs);
}
/*! \brief operator overload */
template<typename TA, int ta>
inline BinaryMapExp<op::div, TA, ScalarExp<MSHADOW_SCALAR_>,
                    MSHADOW_SCALAR_, (ta|type::kMapper)>
operator/(const Exp<TA, MSHADOW_SCALAR_, ta> &lhs,
          const MSHADOW_SCALAR_ &rhs) {
  return MakeExp<op::div>(lhs, scalar<MSHADOW_SCALAR_>(rhs));
}
// constant operators 2
/*! \brief operator overload */
template<typename TB, int tb>
inline BinaryMapExp<op::plus, ScalarExp<MSHADOW_SCALAR_>, TB,
                    MSHADOW_SCALAR_, (tb|type::kMapper)>
operator+(MSHADOW_SCALAR_ lhs,
          const Exp<TB, MSHADOW_SCALAR_, tb> &rhs) {
  return MakeExp<op::plus>(scalar<MSHADOW_SCALAR_>(lhs), rhs);
}
/*! \brief operator overload */
template<typename TB, int tb>
inline BinaryMapExp<op::minus, ScalarExp<MSHADOW_SCALAR_>, TB,
                    MSHADOW_SCALAR_, (tb|type::kMapper)>
operator-(MSHADOW_SCALAR_ lhs,
          const Exp<TB, MSHADOW_SCALAR_, tb> &rhs) {
  return MakeExp<op::minus>(scalar<MSHADOW_SCALAR_>(lhs), rhs);
}
/*! \brief operator overload */
template<typename TB, int tb>
inline BinaryMapExp<op::mul, ScalarExp<MSHADOW_SCALAR_>, TB,
                    MSHADOW_SCALAR_, (tb|type::kMapper)>
operator*(MSHADOW_SCALAR_ lhs,
          const Exp<TB, MSHADOW_SCALAR_, tb> &rhs) {
  return MakeExp<op::mul>(scalar<MSHADOW_SCALAR_>(lhs), rhs);
}
/*! \brief operator overload */
template<typename TB, int tb>
inline BinaryMapExp<op::div, ScalarExp<MSHADOW_SCALAR_>, TB,
                    MSHADOW_SCALAR_, (tb|type::kMapper)>
operator/(MSHADOW_SCALAR_ lhs, const Exp<TB, MSHADOW_SCALAR_, tb> &rhs) {
  return MakeExp<op::div>(scalar<MSHADOW_SCALAR_>(lhs), rhs);
}
}  // namespace expr
}  // namespace mshadow
