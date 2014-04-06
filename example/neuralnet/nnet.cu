// this implements a simple two layer neural net
#include <vector>
// header file to use mshadow
#include "mshadow/tensor.h"
// helper function to load mnist dataset
#include "util.h"
// this namespace contains all data structures, functions
using namespace mshadow;
// this namespace contains all operator overloads
using namespace mshadow::expr;

// define sigmoid operation
struct sigmoid{
    MSHADOW_XINLINE static real_t Map(real_t a) {
        return  1.0f/(1.0f+expf(-a));
    }
};

/*! \brief interface for nnet, interfacd allows use to use GPU/CPU implementation in a unified way */
class INNet{
public:
    virtual void Forward( const Tensor<cpu,2>& inbatch, Tensor<cpu,2> &oubatch ) = 0;
    virtual void Backprop( const Tensor<cpu,2>& gradout ) = 0;    
    virtual void Update( void ) = 0;
    virtual ~INNet(){}
};

/*! 
 * \brief simple two layer neural net 
 *        this implementation is device invariant
 */
template<typename xpu>
class NNet : public INNet{
public:
    // initialize the network
    NNet( int batch_size, int num_in, int num_hidden, int num_out ):rnd(0){
        // setup nodes
        ninput.Resize( Shape2( batch_size, num_in ) );
        nhidden.Resize( Shape2( batch_size, num_hidden ) );
        nhiddenbak.Resize( nhidden.shape );
        nout.Resize( Shape2( batch_size, num_out ) );
        // setup bias
        hbias.Resize( Shape1( num_hidden ) ); g_hbias.Resize( hbias.shape );
        obias.Resize( Shape1( num_out ) ); g_obias.Resize( obias.shape );
        hbias = 0.0f; obias = 0.0f;
        // setup weights
        Wi2h.Resize( Shape2( num_in, num_hidden ) );  g_Wi2h.Resize( Wi2h.shape );
        Wh2o.Resize( Shape2( num_hidden, num_out ) ); g_Wh2o.Resize( Wh2o.shape );
        rnd.SampleGaussian( Wi2h, 0, 0.01f );
        rnd.SampleGaussian( Wh2o, 0, 0.01f );

    }
    virtual ~NNet(){}
    // forward propagation
    virtual void Forward( const Tensor<cpu,2>& inbatch, Tensor<cpu,2> &oubatch ){
        // note: in mshadow, shape[0] means lowest dimension, shape[1] is number of rows in matrix
        // this is different from numpy convention
        index_t batch_size = inbatch.shape[1];
        // copy data to input layer
        Copy( ninput, inbatch );
        // first layer, fullc
        nhidden = dot( ninput, Wi2h );
        nhidden+= repmat( hbias, batch_size );
        // activation, sigmloid, backup activation in nhidden 
        nhidden = F<sigmoid>( nhidden );
        Copy( nhiddenbak, nhidden );
        // second layer fullc
        nout = dot( nhiddenbak, Wh2o );
        nout += repmat( obias, batch_size );
        // softmax calculation
        Softmax( nout, nout );
        // copy result out
        Copy( oubatch, nout );
    }
    // back propagation
    virtual void Backprop( const Tensor<cpu,2>& gradout ){        
        // copy gradient to output layer
        Copy( nout, gradout );
        // calc grad of layer 2
        g_obias = sum_rows( nout );
        g_Wh2o  = dot( nhiddenbak.T(), nout );
        // backprop to layer 1 
        nhiddenbak = dot( nout, Wh2o.T() );
        // calculate gradient of sigmoid layer
        nhidden = nhidden * (1.0f-nhidden) * nhiddenbak;
        // calc grad of layer 1
        g_hbias = sum_rows( nhidden );
        g_Wi2h  = dot( ninput.T(), nhidden );        
    }
    // update weight
    virtual void Update( void ){
        // run SGD
        const float eta = 0.8;
        const float wd = 0.00001;
        // update weight
        Wi2h -= eta * ( wd * Wi2h + g_Wi2h );
        Wh2o -= eta * ( wd * Wh2o + g_Wh2o );
        // no regularization for bias
        hbias-= eta * g_hbias;
        obias-= eta * g_obias;        
    }
private:
    // random seed generator
    Random<xpu> rnd;
    // nodes in neural net
    TensorContainer<xpu,2> ninput, nhidden, nhiddenbak, nout;
    // hidden bias, gradient
    TensorContainer<xpu,1> hbias, obias, g_hbias, g_obias;
    // weight gradient
    TensorContainer<xpu,2> Wi2h, Wh2o, g_Wi2h, g_Wh2o;    
};

// helper function to get the max inde
inline int MaxIndex( Tensor<cpu,1> pred ){
    int maxidx = 0;
    for( index_t i = 1; i < pred.shape[0]; ++i ){
        if( pred[i] > pred[maxidx] ) maxidx = (int)i;
    }
    return maxidx;
}

int main( int argc, char *argv[] ){
    if( argc < 2 ){
        printf("Usage: cpu or gpu\n"); return 0;
    }
    srand(0);
    InitTensorEngine();

    // settings
    int batch_size = 100;
    int num_in = 28 * 28;
    int num_hidden = 100;
    int num_out = 10;

    // choose which version to use
    INNet *net;
    if( !strcmp( argv[1], "gpu") ) {
        net = new NNet<gpu>( batch_size, num_in, num_hidden, num_out );
    }else{
        net = new NNet<cpu>( batch_size, num_in, num_hidden, num_out );
    }

    // temp output layer
    TensorContainer<cpu,2> pred;    
    pred.Resize( Shape2( batch_size, num_out ) );
    
    // label 
    std::vector<int> ytrain, ytest;
    // data
    TensorContainer<cpu,2> xtrain, xtest;
    LoadMNIST( "train-images-idx3-ubyte", "train-labels-idx1-ubyte", ytrain, xtrain, true);
    LoadMNIST( "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", ytest, xtest, false);
    
    int num_iter = 20;

    for( int i = 0; i < num_iter; ++ i ){
        // training 
        for( index_t j = 0; j + batch_size <= xtrain.shape[1]; j += batch_size ){
            net->Forward( xtrain.Slice( j, j + batch_size ), pred );
            // set gradient into pred
            for( int k = 0; k < batch_size; ++ k ){
                pred[k][ ytrain[k+j] ] -= 1.0f;
            }
            // scale gradient by batchs zie
            pred *= 1.0f / batch_size;
            // run backprop
            net->Backprop( pred );
            // update net parameters
            net->Update();
        }
        // evaluation
        long nerr = 0;
        for( index_t j = 0; j + batch_size <= xtest.shape[1]; j += batch_size ){
            net->Forward( xtest.Slice( j, j + batch_size ), pred );            
            for( int k = 0; k < batch_size; ++ k ){                
                nerr += MaxIndex( pred[k] ) != ytest[j+k];
                
            }
        }
        printf("round %d: test-err=%f\n", i, (float)nerr/xtest.shape[1] );
    }    
    delete net;
    ShutdownTensorEngine();
    return 0;
}
