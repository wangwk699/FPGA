// Small compile-only probe for a relocated Vitis HLS installation; no card load.
extern "C" void shared_toolchain_probe(const float *input, float *output, int count) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=read_data
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=write_data
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=count bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    for (int i = 0; i < count; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = input[i] + 1.0f;
    }
}
