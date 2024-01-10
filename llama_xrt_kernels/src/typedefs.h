#include <stdint.h>
#include <stdio.h>

// TODO: replace with HLS types (vector, int, fp, etc...)

//===========================================================================
//  typedefs.h
//===========================================================================
//  @brief: This header defines the shorthand of several ap_uint data types.

#ifndef TYPEDEFS
#define TYPEDEFS

// typedef bool bit;
// typedef ap_int<8> bit8_t;
// typedef ap_int<16> bit16_t;
// typedef ap_uint<2> bit2_t;
// typedef ap_uint<4> bit4_t;
// typedef ap_uint<32> bit32_t;

struct Config
{
  int dim;        // transformer dimension
  int hidden_dim; // for ffn layers
  int n_layers;   // number of layers
  int n_heads;    // number of query heads
  int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
  int vocab_size; // vocabulary size, usually 256 (byte-level)
  int seq_len;    // max sequence length
  int GS;         // group size
};

template <int SIZE>
struct QuantizedTensor
{
  int8_t q[SIZE]; // quantized values
  float s[SIZE];  // scaling factors
};


template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
struct TransformerWeights
{
  // token embedding table
  QuantizedTensor<vocab_size * dim> q_tokens[1]; // (vocab_size, dim)
  float token_embedding_table[vocab_size * dim]; // same, but dequantized

  // weights for rmsnorms
  float rms_att_weight[n_layers * dim]; // (layer, dim) rmsnorm weights
  float rms_ffn_weight[n_layers * dim]; // (layer, dim)
  // weights for matmuls. note dim == n_heads * head_size
  QuantizedTensor<dim *(dim / n_heads) * n_heads> wq[n_layers];    // (layer, dim, n_heads * head_size)
  QuantizedTensor<dim *(dim / n_heads) * n_kv_heads> wk[n_layers]; // (layer, dim, n_kv_heads * head_size)
  QuantizedTensor<dim *(dim / n_heads) * n_kv_heads> wv[n_layers]; // (layer, dim, n_kv_heads * head_size)
  QuantizedTensor<n_heads * dim *(dim / n_heads)> wo[n_layers];    // (layer, dim, n_heads * head_size)
  // weights for ffn
  QuantizedTensor<dim * hidden_dim> w1[n_layers]; // (layer, hidden_dim, dim)
  QuantizedTensor<dim * hidden_dim> w2[n_layers]; // (layer, dim, hidden_dim)
  QuantizedTensor<dim * hidden_dim> w3[n_layers]; // (layer, hidden_dim, dim)
  // final rmsnorm
  float rms_final_weight[dim]; // (dim,)
  // (optional) classifier weights for the logits, on the last layer
  QuantizedTensor<vocab_size * dim> wcls[1];
};

// ----------------------------------------------------------------------------
// Transformer model

template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
struct Transformer
{
  Config config;                                                                                       // the hyperparameters of the architecture (the blueprint)
  TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> weights; // the weights of the model
  // some more state needed to properly clean up the memory mapping (sigh)
  // int fd;            // file descriptor for memory mapping
  // float *data;       // memory mapped data pointer
  // ssize_t file_size; // size of the checkpoint file in bytes
};
#endif
