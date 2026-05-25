module matrix_mult_4x4x4x16 (
    input wire clk,
    input wire reset,
    input wire enable,
    input wire signed [15:0] activations [0:3][0:3],
    input wire signed [15:0] weights [0:3][0:3],
    output wire signed [31:0] results [0:3][0:3]
);

matrix_vector_mult_4x4x16 mxv0 (
    .clk(clk),
    .reset(reset),
    .enable(enable),
    ----------------------
    ----------------------
    ----------------------
);
matrix_vector_mult_4x4x16 mxv1 (
    .clk(clk),
    .reset(reset),
    .enable(enable),
    ----------------------
    ----------------------
    ----------------------
);
matrix_vector_mult_4x4x16 mxv2 (
    .clk(clk),
    .reset(reset),
    .enable(enable),
    ----------------------
    ----------------------
    ----------------------
);
matrix_vector_mult_4x4x16 mxv3 (
    .clk(clk),
    .reset(reset),
    .enable(enable),
    ----------------------
    ----------------------
    ----------------------
);

endmodule
