#include "forward.h"
#include "config.h"
#include <cstring>

// TODO: include HLS math package

// neural net blocks; the dynamics of the Transformer
template <int S>

// o = output, x = input, weight = array of gain params

// CHECKED: RMSNorm seems good
void rmsnorm(float o[S], float x[S], float weight[S])
{
  constexpr auto array_size = S * sizeof(float);
  // calculate sum of squares
  float ss = 0.0f;
  float x_buff[S];
  float weight_buff[S];
  float out_buff[S];
#pragma HLS array_partition variable = x_buff type = cyclic factor = 128
#pragma HLS array_partition variable = weight_buff type = cyclic factor = 64
#pragma HLS array_partition variable = out_buff type = cyclic factor = 64
  std::memcpy(x_buff, x, array_size);
  std::memcpy(weight_buff, weight, array_size);

sum_of_squares:
  for (int j = 0; j < S; j++)
  {
#pragma HLS PIPELINE
#pragma HLS UNROLL factor = 128 skip_exit_check
    float x_j = x_buff[j];
    ss += x_j * x_j;
  }
  ss /= S;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
// normalize and scale
norm_and_scale:
  for (int j = 0; j < S; j++)
  {
#pragma HLS PIPELINE
#pragma HLS UNROLL factor = 64
    float weight_j = weight_buff[j];
    float x_j = x_buff[j];
    out_buff[j] = weight_j * (ss * x_j);
  }
  std::memcpy(o, out_buff, array_size);
}

template <int MAXSIZE>
void softmax(float *x, int size)
{
  // find max value (for numerical stability)
  float buffer[MAXSIZE];
  float max_val = x[0];
max:
  for (int i = 1; i < size; i++)
  {
#pragma HLS loop_tripcount min = 0 max = 257 avg = 129
#pragma HLS PIPELINE
    float x_i = x[i];
    if (x_i > max_val)
    {
      max_val = x_i;
    }
  }
  // exp and sum

exp:
  for (int i = 0; i < size; i++)
  {
#pragma HLS loop_tripcount min = 0 max = 257 avg = 129
#pragma HLS PIPELINE
#pragma HLS UNROLL factor = 16
    float x_i = expf(x[i] - max_val);
    buffer[i] = x_i;
  }
  float sum = 0.0f;
sum:
  for (int i = 0; i < size; i++)
  {
#pragma HLS loop_tripcount min = 0 max = 257 avg = 129
    sum += buffer[i];
  }
  // normalize
  const float inv_sum = 1.0 / sum;
norm:
  for (int i = 0; i < size; i++)
  {
#pragma HLS loop_tripcount min = 0 max = 257 avg = 129
#pragma HLS PIPELINE
#pragma HLS UNROLL factor = 16
    x[i] = buffer[i] * inv_sum;
  }
}

template <int N, int D>
void matmul_old(float *xout, int8_t *xq, float *xs, int8_t *wq, float *ws)
{
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  // inputs to this function are both quantized

  // wq - quantized weight matrix
  // ws - scaling factor for each row of wq
  // xq - quantized input vector
  // xs - scaling factor for xq
  // xout - output vector

  static int8_t x_buffer[N];
  static float xs_buffer[N / GS];
  // float out_buffer[D];
  int8_t w_buffer[N * D];
  float ws_buffer[N * D / GS];

#pragma HLS ARRAY_PARTITION variable = x_buffer type = cyclic factor = 16
#pragma HLS ARRAY_PARTITION variable = xs_buffer type = cyclic factor = 4
#pragma HLS ARRAY_PARTITION variable = w_buffer type = cyclic factor = 128
#pragma HLS ARRAY_PARTITION variable = ws_buffer type = cyclic factor = 32
//
x_buff:
  for (int i = 0; i < N; i++)
  {
#pragma HLS UNROLL factor = 16
    x_buffer[i] = xq[i];
  }
xs_buff:
  for (int j = 0; j <= N - GS; j += GS)
  {
#pragma HLS UNROLL factor = 4
    xs_buffer[j / GS] = xs[j / GS];
  }

w_buff:
  for (int i = 0; i < N * D; i++)
  {
#pragma HLS UNROLL factor = 128
    w_buffer[i] = wq[i];
  }

ws_buff:
  for (int i = 0; i < N * D / GS; i++)
  {
#pragma HLS UNROLL factor = 32
    ws_buffer[i] = ws[i];
  }

  int i;
  for (i = 0; i < D; i++)
  {
#pragma HLS PIPEPLINE
    float val = 0.0f;
    // start index of row i
    const int in = i * N;
    // matmul1:
    // for (int j = 0; j < N; j++)
    //{
    // #pragma HLS UNROLL
    //  w_buffer[j] = wq[j + in];
    //}
    // matmul2:
    const int in_s = i * N / GS;
    // const int groups = N / GS;
    // for (int j = 0; j < groups; j++)
    //  {
    // #pragma HLS UNROLL
    //  //  ws_buffer[j] = ws[in_s + j];
    // }

    // do the matmul in groups of GS

    int j;
  matmul3:
    for (j = 0; j <= N - GS; j += GS)
    {
#pragma HLS UNROLL
      int32_t ival = 0;
    matmul4:
      for (int k = 0; k < GS; k++)
      {
#pragma HLS UNROLL
        ival += ((int32_t)x_buffer[j + k]) * ((int32_t)w_buffer[in + j + k]);
      }
      val += ((float)ival) * ws_buffer[in_s + j / GS] * xs_buffer[j / GS];
    }
    xout[i] = val;
  }
}

template <int N, int D>
void matmul(float *xout, int8_t *xq, float *xs, int8_t *wq, float *ws)
{
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  // inputs to this function are both quantized

  // wq - quantized weight matrix
  // ws - scaling factor for each row of wq
  // xq - quantized input vector
  // xs - scaling factor for xq
  // xout - output vector

  static int8_t x_buffer[N];
  static float xs_buffer[N / GS];
  // float out_buffer[D];

#pragma HLS ARRAY_PARTITION variable = x_buffer type = cyclic factor = 16
#pragma HLS ARRAY_PARTITION variable = xs_buffer type = cyclic factor = 4
//
x_buff:
  for (int i = 0; i < N; i++)
  {
#pragma HLS UNROLL factor = 16
    x_buffer[i] = xq[i];
  }
xs_buff:
  for (int j = 0; j <= N - GS; j += GS)
  {
#pragma HLS UNROLL factor = 4
    xs_buffer[j / GS] = xs[j / GS];
  }

  int i;
  for (i = 0; i < D; i++)
  {
#pragma HLS PIPELINE
    float val = 0.0f;
    int8_t w_buffer[N];
    float ws_buffer[N / GS];
#pragma HLS ARRAY_PARTITION variable = w_buffer type = cyclic factor = 32
#pragma HLS ARRAY_PARTITION variable = ws_buffer type = cyclic factor = 32
    // start index of row i
    const int in = i * N;
  matmul1:
    for (int j = 0; j < N; j++)
    {
      // #pragma HLS UNROLL factor
      w_buffer[j] = wq[j + in];
    }
  matmul2:
    const int in_s = i * N / GS;
    const int groups = N / GS;
    for (int j = 0; j < groups; j++)
    {
      // #pragma HLS UNROLL factor
      ws_buffer[j] = ws[in_s + j];
    }

    // do the matmul in groups of GS
    int j;
  matmul3:
    for (j = 0; j <= N - GS; j += GS)
    {
      // #pragma HLS UNROLL
      int32_t ival = 0;
    matmul4:
      for (int k = 0; k < GS; k++)
      {
        // #pragma HLS UNROLL
        ival += ((int32_t)x_buffer[j + k]) * ((int32_t)w_buffer[j + k]);
      }
      val += ((float)ival) * ws_buffer[j / GS] * xs_buffer[j / GS];
    }
    xout[i] = val;
  }
}

extern "C" void forward(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *transformer, int token, int pos, float key_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)], float value_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)], float *out)
{
#pragma HLS INTERFACE m_axi port = transformer offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem1

  // a few convenience variables
  auto w = &transformer->weights;
  constexpr int UNROLL_FACTOR = 16;
  static float x[config.dim];                                        // activation at current time stamp (dim,)
  static float xb[config.dim];                                       // same, but inside a residual branch (dim,)
  static float xb2[config.dim];                                      // an additional buffer just for convenience (dim,)
  static float hb[config.hidden_dim];                                // buffer for hidden dimension in the ffn (hidden_dim,)
  static float hb2[config.hidden_dim];                               // buffer for hidden dimension in the ffn (hidden_dim,)
  static QuantizedTensor<config.dim> xq;                             // quantized x (dim,)
  static QuantizedTensor<config.hidden_dim> hq;                      // quantized hb (hidden_dim,)
  static float q[config.dim];                                        // query (dim,)
  static float k[(config.dim * config.n_kv_heads) / config.n_heads]; // key (dim,)
  static float v[(config.dim * config.n_kv_heads) / config.n_heads]; // value (dim,)
  static float att[config.n_heads * config.seq_len];                 // buffer for scores/attention values (n_heads, seq_len)
#pragma HLS ARRAY_PARTITION variable = q cyclic factor = UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = k cyclic factor = UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = v cyclic factor = UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = att cyclic factor = UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = hq.q cyclic factor = UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = hq.s cyclic factor = UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = xq.q cyclic factor = UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = xq.s cyclic factor = UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = hb type = cyclic factor = UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = hb2 type = cyclic factor = UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = x type = cyclic factor = UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = xb type = cyclic factor = UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = xb2 type = cyclic factor = UNROLL_FACTOR
  constexpr int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
  constexpr int kv_mul = config.n_heads / config.n_kv_heads; // integer multiplier of the kv sharing in multiquery
  constexpr int head_size = dim / config.n_heads;

  // copy the token embedding into x
  std::memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(float));

// forward all the layers
main_forward_loop:
  for (int l = 0; l < config.n_layers; l++)
  {
    // attention rmsnorm
    rmsnorm<dim>(xb, x, w->rms_att_weight + l * dim);

    // qkv matmuls for this position
    quantize(&xq, xb, GS);
    matmul<dim, dim>(q, xq.q, xq.s, (w->wq + l)->q, (w->wq + l)->s);
    matmul<dim, kv_dim>(k, xq.q, xq.s, (w->wk + l)->q, (w->wk + l)->s);
    matmul<dim, kv_dim>(v, xq.q, xq.s, (w->wv + l)->q, (w->wv + l)->s);

  // RoPE relative positional encoding: complex-valued rotate q and k in each head
  // Process the portion where both query and key vectors are involved (i < kv_dim)
  rotation1:
    // Rotation for both query and key vectors (i < kv_dim)
    for (int i = 0; i < kv_dim; i += 2)
    {
#pragma HLS UNROLL factor = UNROLL_FACTOR
#pragma HLS PIPELINE
      int head_dim = i % head_size;
      float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
      float val = pos * freq;
      float fcr = cosf(val);
      float fci = sinf(val);

      // Rotate the query vector
      float v0_q = q[i];
      float v1_q = q[i + 1];
      q[i] = v0_q * fcr - v1_q * fci;
      q[i + 1] = v0_q * fci + v1_q * fcr;

      // Rotate the key vector
      float v0_k = k[i];
      float v1_k = k[i + 1];
      k[i] = v0_k * fcr - v1_k * fci;
      k[i + 1] = v0_k * fci + v1_k * fcr;
    }
  rotation2:
    // Rotation for only the query vector (i >= kv_dim)
    for (int i = kv_dim; i < dim; i += 2)
    {
#pragma HLS PIPELINE
      int head_dim = i % head_size;
      float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
      float val = pos * freq;
      float fcr = cosf(val);
      float fci = sinf(val);

      // Rotate only the query vector
      float v0 = q[i];
      float v1 = q[i + 1];
      q[i] = v0 * fcr - v1 * fci;
      q[i + 1] = v0 * fci + v1 * fcr;
    }
    // save key,value at this time step (pos) to our kv cache
    int loff = l * config.seq_len * kv_dim; // kv cache layer offset for convenience
    float *key_cache_row = key_cache + loff + pos * kv_dim;
    float *value_cache_row = value_cache + loff + pos * kv_dim;

    std::memcpy(key_cache_row, k, kv_dim * sizeof(*key_cache_row));
    std::memcpy(value_cache_row, v, kv_dim * sizeof(*value_cache_row));

    // multihead attention. iterate over all heads
    int h;

  multihead_attention:
    for (h = 0; h < n_heads; h++)
    {
      //  get the query vector for this head
      // float *q_t = q + h * head_size;
      const int q_offset = h * head_size;
      // attention scores for this head
      // float *att_t = att + h * seq_len;
      const int att_offset = h * seq_len;
    // iterate over all timesteps, including the current one
    iterate:
      for (int t = 0; t <= pos; t++)
      {
#pragma HLS PIPELINE
#pragma HLS loop_tripcount min = 0 max = 257 avg = 129
        // get the key vector for this head and at this timestep
        // float *k_t = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        const int key_offset = loff + t * kv_dim + (h / kv_mul) * head_size;
        // calculate the attention score as the dot product of q and k
        float score = 0.0f;
        for (int i = 0; i < head_size; i++)
        {
#pragma HLS unroll
          score += q[i + q_offset] * key_cache[i + key_offset];
        }
        score /= sqrtf(head_size);
        // save the score to the attention buffer
        att[t + att_offset] = score;
      }

      // softmax the scores to get attention weights, from 0..pos inclusively
      softmax<257>(att + att_offset, pos + 1);

      // weighted sum of the values, store back into xb
      // float *xb_t = xb + h * head_size;
      const int xb_offset = h * head_size;
      memset(xb + xb_offset, 0, head_size * sizeof(float));
    acc:
      for (int t = 0; t <= pos; t++)
      {
#pragma HLS loop_tripcount min = 0 max = 257 avg = 129
#pragma HLS PIPELINE
        // get the value vector for this head and at this timestep
        // float *v_t = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        // get the attention weight for this timestep
        const int v_offset = loff + t * kv_dim + (h / kv_mul) * head_size;
        float a = att[t + att_offset];
      // accumulate the weighted value into xb
      acc_inner:
        for (int i = 0; i < head_size; i++)
        {
#pragma HLS unroll
          xb[i + xb_offset] += a * value_cache[i + v_offset];
        }
      }
    }

    // final matmul to get the output of the attention
    quantize(&xq, xb, GS);
    matmul<dim, dim>(xb2, xq.q, xq.s, (w->wo + l)->q, (w->wo + l)->s);

  // residual connection back into x
  residual:
    for (int i = 0; i < dim; i++)
    {
#pragma HLS UNROLL factor = 64 skip_exit_check
      x[i] += xb2[i];
    }

    // ffn rmsnorm
    rmsnorm<dim>(xb, x, w->rms_ffn_weight + l * dim);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    quantize(&xq, xb, GS);
    matmul<dim, hidden_dim>(hb, xq.q, xq.s, (w->w1 + l)->q, (w->w1 + l)->s);
    matmul<dim, hidden_dim>(hb2, xq.q, xq.s, (w->w3 + l)->q, (w->w3 + l)->s);
    float hb_out[hidden_dim];
#pragma HLS array_partition variable = hb_out type = cyclic factor = 16
  swi_glu:
    for (int i = 0; i < hidden_dim; i++)
    {
#pragma HLS UNROLL factor = 4
#pragma HLS PIPELINE
      float val = hb[i];
      // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
      val *= (1.0f / (1.0f + expf(-val)));
      // elementwise multiply with w3(x)
      val *= hb2[i];
      hb_out[i] = val;
    }
    std::memcpy(hb, hb_out, hidden_dim * sizeof(float));

    // final matmul to get the output of the ffn
    quantize(&hq, hb, GS);
    matmul<hidden_dim, dim>(xb, hq.q, hq.s, (w->w2 + l)->q, (w->w2 + l)->s);

  // residual connection
  residual2:
    for (int i = 0; i < dim; i++)
    {
#pragma HLS UNROLL factor = 16 skip_exit_check
      x[i] += xb[i];
    }
  }

  // final rmsnorm
  rmsnorm<dim>(x, x, w->rms_final_weight);

  // classifier into logits
  quantize(&xq, x, GS);
  matmul<dim, vocab_size>(out, xq.q, xq.s, w->wcls->q, w->wcls->s);
}
