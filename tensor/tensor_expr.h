#ifndef CXXNET_TENSOR_EXPR_H
#define CXXNET_TENSOR_EXPR_H
/*!
 * \file tensor_expr.h
 * \brief definitions of abstract expressions and expressions template
 * \author Tianqi Chen
 */
#include "tensor_op.h"

namespace cxxnet{
    /*! 
     * \brief namespace for abstract expressions and expressions template,
     *        have no dependecy on tensor.h, 
     *        These data structure takes no charge in computations, 
     *        they are only used to define operations and represent expression in a symbolic way
     */
    namespace expr{
        
        /*! \brief type of expressions */
        namespace type{
            /*! \brief this expression directly correspnds to a data class */
            const int kContainer = 0;
            /*! \brief this only contains element-wise vector operations */
            const int kMapper    = 1;
            /*! \brief othercase: e.g dot product */
            const int kComplex   = 3;
        };

        /*! 
         * \brief expression engine that actually interprets these expressions 
         *        this is a function template that needed to be implemented for specific expressions
         */
        template<typename Saver,typename Container>
        struct ExpEngine{
            template<typename EType>
            inline static void eval( Container& dst, const EType &exp );
        };


        template<typename E>
        class TransposeExp;
        template<typename Container>
        class ContainerExp;
        class ScalarExp;

        /*! 
         * \brief base class for expression 
         * \tparam SubType inheritated class must put their type into this parameter
         * \tparam exp_type expression type, see namespace type
         */
        template<typename SubType, int exp_type>
        struct Exp{
        public:
            typedef SubType EType;
            const static int kType = exp_type;
        public:
            inline const SubType& self( void ) const{
                return *static_cast<const SubType*>(this);
            }
            inline SubType& refself( void ){
                return *static_cast<const SubType*>(this);
            }
            /*! 
             *\brief transpose of a matrix
             *\return transpose of current expression
             */
            inline const TransposeExp<EType> T( void ) const{
                return TransposeExp<EType>( self() );
            }
        };
                  
        /*! \brief transpose of a expression*/
        template<typename EType>
        struct TransposeExp: public Exp< TransposeExp<EType>, type::kComplex >{
            /*! \brief expression to be transposed */
            const EType &exp;
            /*! \brief constructor */
            TransposeExp( const EType &e ):exp(e){}        
            inline const EType & T( void ) const{
                return exp;
            }
        };

        /*! \brief scalar expression */
        struct ScalarExp: public Exp<ScalarExp, type::kMapper>{
            real_t scalar_;
            ScalarExp( real_t scalar ):scalar_(scalar){}
            ScalarExp( double scalar ):scalar_(scalar){}
        };
        
        /*! 
         * \brief base class of all variables, that can be assigned to values
         * \tparam Container the actually class of data container, e.g. CTensor1D
         */
        template<typename Container>
        class ContainerExp: public Exp< Container, type::kContainer >{
        public:
            inline Container &operator+=( double s ){
                ExpEngine<sv::plusto,Container>::eval( this->refself(), ScalarExp(s) );
                return this->refself();
            }
            inline Container &operator-=( double s ){
                ExpEngine<sv::minusto,Container>::eval( this->refself(), ScalarExp(s) );
                return this->refself();
            }
            inline Container &operator*=( double s ){
                ExpEngine<sv::multo,Container>::eval( this->refself(), ScalarExp(s) );
                return this->refself();
            }
            inline Container &operator/=( double s ){
                ExpEngine<sv::divto,Container>::eval( this->refself(), ScalarExp(s) );
                return this->refself();
            }
            inline Container &__assign( double s ){
                ExpEngine<sv::saveto,Container>::eval( this->refself(), ScalarExp(s) );
                return this->refself();
            }
        public:                
            /*! \brief implementation of operator=, note that we can not define container = container */
            template<typename E>
            inline Container &__assign( const Exp<E,type::kMapper> &exp ){
                ExpEngine<sv::saveto,Container>::eval( this->refself(), exp.self() );
                return this->refself();
            }
            template<typename E>
            inline Container &__assign( const Exp<E,type::kComplex> &exp ){
                ExpEngine<sv::saveto,Container>::eval( this->refself(), exp.self() );
                return this->refself();
            }
            /*! \brief implementation of operator+= */
            template<typename E,int etype>
            inline Container &operator+=( const Exp<E,etype> &exp ){
                ExpEngine<sv::plusto,Container>::eval( this->refself(), exp.self() );
                return this->refself();
            }
            /*! \brief implementation of operator-= */
            template<typename E,int etype>
            inline Container &operator-=( const Exp<E,etype> &exp ){
                ExpEngine<sv::minusto,Container>::eval( this->refself(), exp.self() );
                return this->refself();
            }
            /*! \brief implementation of operator*= */
            template<typename E,int etype>
            inline Container &operator*=( const Exp<E,etype> &exp ){
                ExpEngine<sv::multo,Container>::eval( this->refself(), exp.self() );
                return this->refself();
            }
            /*! \brief implementation of operator/= */
            template<typename E,int etype>
            inline Container &operator/=( const Exp<E,etype> &exp ){
                ExpEngine<sv::divto,Container>::eval( this->refself(), exp.self() );
                return this->refself();
            }
        };
    };

    namespace expr{
        /*! 
         * \brief binary map expression lhs [op] rhs
         * \tparam OP operator
         * \tparam TA type of lhs
         * \tparam TB type of rhs
         */
        template<typename OP, typename TA, typename TB, int etype >
        struct BinaryMapExp: public Exp< BinaryMapExp<OP,TA,TB,etype>, etype >{
            const TA &lhs_;
            const TB &rhs_;
            BinaryMapExp( const TA &lhs, const TB &rhs )
                :lhs_(lhs.self()), rhs_(rhs.self()){}
        };

        // make expression
        template<typename OP,typename TA, typename TB, int ta, int tb>
        inline BinaryMapExp<OP,TA,TB, (ta|tb|type::kMapper) >  MakeExp( const Exp<TA,ta> &lhs, const Exp<TB,tb> &rhs ){
            return BinaryMapExp<OP,TA,TB, (ta|tb|type::kMapper) >( lhs.self(), rhs.self() );
        }

        // operator rules
        template<typename TA, typename TB, int ta, int tb>
        inline BinaryMapExp<op::plus,TA,TB, (ta|tb|type::kMapper) > operator+( const Exp<TA,ta> &lhs, const Exp<TB,tb> &rhs ){
            return MakeExp<op::plus>( lhs.self(), rhs.self() );
        }
        template<typename TA, typename TB, int ta, int tb>
        inline BinaryMapExp<op::minus,TA,TB, (ta|tb|type::kMapper) > operator-( const Exp<TA,ta> &lhs, const Exp<TB,tb> &rhs ){
            return MakeExp<op::minus>( lhs.self(), rhs.self() );
        }
        template<typename TA, typename TB, int ta, int tb>
        inline BinaryMapExp<op::mul,TA,TB, (ta|tb|type::kMapper) > operator*( const Exp<TA,ta> &lhs, const Exp<TB,tb> &rhs ){
            return MakeExp<op::mul>( lhs.self(), rhs.self() );
        }
        template<typename TA, typename TB, int ta, int tb>
        inline BinaryMapExp<op::div,TA,TB, (ta|tb|type::kMapper) > operator/( const Exp<TA,ta> &lhs, const Exp<TB,tb> &rhs ){
            return MakeExp<op::div>( lhs.self(), rhs.self() );
        }
        // constant operators 
        template<typename TA, int ta>
        inline BinaryMapExp<op::plus, TA, ScalarExp, (ta|type::kMapper) > operator+( const Exp<TA,ta>& lhs,  real_t rhs ){
            return MakeExp<op::plus>( lhs.self(), ScalarExp(rhs) );
        }
        template<typename TA, int ta>
        inline BinaryMapExp<op::plus, TA, ScalarExp, (ta|type::kMapper) > operator-( const Exp<TA,ta>& lhs,  real_t rhs ){
            return MakeExp<op::plus>( lhs.self(), ScalarExp(-rhs) );
        }
        template<typename TA, int ta>
        inline BinaryMapExp<op::mul, TA, ScalarExp, (ta|type::kMapper) > operator*( const Exp<TA,ta>& lhs,  real_t rhs ){
            return MakeExp<op::mul>( lhs.self(), ScalarExp(rhs) );
        }
        template<typename TA, int ta>
        inline BinaryMapExp<op::div, TA, ScalarExp, (ta|type::kMapper) > operator/( const Exp<TA,ta>& lhs,  real_t rhs ){
            return MakeExp<op::div>( lhs.self(), ScalarExp(rhs) );
        }
        // constant operators 2
        template<typename TB, int tb>
        inline BinaryMapExp<op::plus, ScalarExp, TB, (tb|type::kMapper) > operator+( real_t lhs, const Exp<TB,tb>& rhs ){
            return MakeExp<op::plus>( ScalarExp(lhs), rhs.self() );
        }
        template<typename TB, int tb>
        inline BinaryMapExp<op::minus, ScalarExp, TB, (tb|type::kMapper) > operator-( real_t lhs, const Exp<TB,tb>& rhs ){
            return MakeExp<op::minus>( ScalarExp(lhs), rhs.self() );
        }
        template<typename TB, int tb>
        inline BinaryMapExp<op::mul, ScalarExp, TB, (tb|type::kMapper) > operator*( real_t lhs, const Exp<TB,tb>& rhs ){
            return MakeExp<op::mul>( ScalarExp(lhs), rhs.self() );
        }
        template<typename TB, int tb>
        inline BinaryMapExp<op::div, ScalarExp, TB, (tb|type::kMapper) > operator/( real_t lhs, const Exp<TB,tb>& rhs ){
            return MakeExp<op::div>( ScalarExp(lhs), rhs.self() );
        }
    };

    namespace expr{
        /*! 
         * \brief unary map expression op(src)
         * \tparam OP operator
         * \tparam TA type of src
         */
        template<typename OP, typename TA, int etype >
        struct UnaryMapExp: public Exp< UnaryMapExp<OP,TA,etype>, etype >{
            const TA &src_;
            UnaryMapExp( const TA &src )
                :src_(src.self()){}
        };
        
        // make expression
        template<typename OP,typename TA, int ta>
        inline UnaryMapExp<OP,TA,(ta|type::kMapper) >  MakeExp( const Exp<TA,ta> &src){
            return UnaryMapExp<OP,TA, (ta|type::kMapper) >( src.self() );
        }
    };
};
#endif
