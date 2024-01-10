#pragma once
#include "typedefs.h"

static constexpr int dim = 768;
static constexpr int hidden_dim = 2048;
static constexpr int n_layers = 12;
static constexpr int n_heads = 12;
static constexpr int n_kv_heads = 12;
static constexpr int vocab_size = 32000;
static constexpr int seq_len = 1024;
static constexpr int GS = 64;

constexpr Config config = {
    .dim = dim,
    .hidden_dim = hidden_dim,
    .n_layers = n_layers,
    .n_heads = n_heads,
    .n_kv_heads = n_kv_heads,
    .vocab_size = vocab_size,
    .seq_len = seq_len,
    .GS = GS,
};
