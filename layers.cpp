#include <stdlib.h>
#include <math.h>
#include "layers.hpp"

Conv_layer::Conv_layer(int in_rows, int in_cols, int in_num_filters, int in_filter_size, float in_learn_rate) {
    rows = in_rows;
    cols = in_cols;
    num_filters = in_num_filters;
    filter_size = in_filter_size;
    learn_rate = in_learn_rate;
}

Conv_layer::Conv_layer()

void Conv_layer::forward(float *dest, float *image, float *filters) {
    int r, c, n, i, j;

    for (n = 0; n < num_filters; n++) {
        for (r = 0; r < rows-2; r++) {
            for (c = 0; c < cols-2; c++) {
                for (i = 0; i < filter_size; i++) {
                    for (j = 0; j < filter_size; j++) {
                        dest[n * ((rows-2) * (cols-2)) + (r * (cols-2) + c)] += image[(r+i) * cols + (c+j)] *
                            filters[(n*filter_size*filter_size) + (i*filter_size + j)];
                    }
                }
            }
        }
    }

    return;
}

void Conv_layer::back(float *dest, float *gradient, float *last_input) {
    float *holder;
    holder = (float *) calloc(filter_size * filter_size, sizeof(float));
    int r, c, n, i, j;

    for (n = 0; n < num_filters; n++) {
        for (i = 0; i < filter_size; i++) {
            for (j = 0; j < filter_size; j++) {
                holder[i * filter_size + j] = 0;
            }
        }
        for (r = 0; r < rows-2; r++) {
            for (c = 0; c < cols-2; c++) {
                for (i = 0; i < filter_size; i++) {
                    for (j = 0; j < filter_size; j++) {
                        holder[i * filter_size + j] += last_input[((r+i)*cols)+(c+j)] *
                            gradient[(n*(rows-2)*(cols-2))+(r*(cols-2)+c)];
                    }
                }
            }
        }
        // Write local var to output var
        for (i = 0; i < filter_size; i++) {
            for (j = 0; j < filter_size; j++) {
                dest[(n*filter_size*filter_size) + (i * filter_size + j)] -=
                    learn_rate * holder[i * filter_size + j];

                //dest[(n*filter_size*filter_size) + (i * filter_size + j)] = holder[i * filter_size + j];
            }
        }
    }

    return;
}


class Maxpool_layer {
    private:
        int rows, cols, num_filters;

    public:
        Maxpool_layer(int in_rows, int in_cols, int in_num_filters) {
            rows = in_rows;
            cols = in_cols;
            num_filters = in_num_filters;
        }
        Maxpool_layer()
        
        void forward(float *dest, float *input) {
            int r, c, n, i, j;
            float holder;

            for (r = 0; r < rows/2; r++) {
                for (c = 0; c < cols/2; c++) {
                    for (n = 0; n < num_filters; n++) {
                        holder = -100.0;
                        for (i = 0; i < 2; i++) {
                            for (j = 0; j < 2; j++) {
                                if (input[(n * rows * cols) + (((r*2)+i)*cols+((c*2)+j))] > holder)
                                    holder = input[(n * rows * cols) + (((r*2)+i)*cols+((c*2)+j))];
                            }
                        }
                        dest[(n * (rows/2) * (cols/2)) + (r*(cols/2) + c)] = holder;
                    }
                }
            }

            return;
        }

        void back(float *dest, float *gradient, float *last_input) {
            int r, c, n, i, j;
            float holder;

            for (r = 0; r < rows/2; r++) {
                for (c = 0; c < cols/2; c++) {
                    for (n = 0; n < num_filters; n++) {
                        holder = -100.0;
                        // find max
                        for (i = 0; i < 2; i++) {
                            for (j = 0; j < 2; j++) {
                                if (last_input[(n * rows * cols) + (((r*2)+i)*cols+((c*2)+j))] > holder)
                                    holder = last_input[(n * rows * cols) + (((r*2)+i)*cols+((c*2)+j))];
                                /*if (r == 0 && c == 2 && n == 0) {
                                    printf("%.15f\t", last_input[(n * rows * cols) + (((r*2)+i)*cols+((c*2)+j))]);
                                }*/
                            }
                        }
                        /*if (r == 0 && c == 2 && n == 0) {
                            printf("\n");
                        }*/

                        // backprop max
                        for (i = 0; i < 2; i++) {
                            for (j = 0; j < 2; j++) {
                                if (last_input[(n * rows * cols) + (((r*2)+i)*cols+((c*2)+j))] == holder)
                                    dest[(n * rows * cols) + ((r*2+i)*cols+(c*2+j))] = gradient[(n*(rows/2)*(cols/2))+(r*(cols/2)+c)];
                            }
                        }
                    }
                }
            }

            return;
        }
};

class Softmax_layer {
    private:
        int in_length, out_length;
        float learn_rate;
    
    public:
        Softmax_layer(int in_in_length, int in_out_length, float in_learn_rate) {
            in_length = in_in_length;
            out_length = in_out_length;
            learn_rate = in_learn_rate;
        }
        Softmax_layer()

        void forward(float *dest, float *input, float *totals, float *weight, float *bias) {
            int i, j;
            float *exp_holder;
            exp_holder = (float *) calloc(10, sizeof(float));
            float sum = 0.0;
            //printf("%0.6f\t%0.6f\t%0.6f\n\n", input[10], weight[100], input[10]*weight[100]);

            for (i = 0; i < 10; i++) {
                for (j = 0; j < 1352; j++) {
                    totals[i] += (float) input[j] * weight[j * 10 + i];
                }
                totals[i] += bias[i];
                exp_holder[i] = exp(totals[i]);
                sum += exp_holder[i];
            }

            for (i = 0; i < out_length; i++) {
                dest[i] = (float) exp_holder[i] / sum;
                //printf("%0.6f\t%0.3f\n", totals[i], bias[i]);
            }

            //printf("\n\n%0.6f\n", weight[101]);
            free(exp_holder);
            return;
        }

        void back(float *dest, float *gradient, float *last_input, float *totals, float *weights, float *bias) {
            int grad_length = out_length;
            int last_input_length = in_length;
            int i, j, a;
            float *exp_holder;
            exp_holder = (float *) calloc(out_length, sizeof(float));
            float sum = 0.0;
            float *d_out_d_t, *d_L_d_t, *d_L_d_w, *d_L_d_inputs;
            d_out_d_t = (float *) calloc(out_length, sizeof(float));
            //float d_t_d_w[1352]; <= last_input
            d_L_d_t = (float *) calloc(out_length, sizeof(float));
            d_L_d_w = (float *) calloc(in_length * out_length, sizeof(float));
            d_L_d_inputs = (float *) calloc(in_length, sizeof(float));

            for (i = 0; i < grad_length; i++) {
                exp_holder[i] = exp(totals[i]);
                sum += exp_holder[i];
            }

            // find index of gradient that != 0
            for (i = 0; i < grad_length; i++) {
                if (gradient[i] != 0) {
                    a = i;
                    break;
                }
            }

            // gradients of out[i] against totals
            for (i = 0; i < grad_length; i++) {
                d_out_d_t[i] = -1* exp_holder[a] * exp_holder[i] / (sum * sum);
            }
            d_out_d_t[a] = exp_holder[a] * (sum - exp_holder[a]) / (sum * sum);

            // gradients of totals against weights/ biases/ input

            // gradients of loss against totals
            for (i = 0; i < grad_length; i++) {
                d_L_d_t[i] = gradient[a] * d_out_d_t[i];
                bias[i] -= learn_rate * d_L_d_t[i];
            }

            // gradients of loss against weights/ biases/ input
            for (i = 0; i < last_input_length; i++) {
                for (j = 0; j < grad_length; j++) {
                    d_L_d_w[i * grad_length + j] = last_input[i] * d_L_d_t[j];
                    d_L_d_inputs[i] += weights[i * grad_length + j] * d_L_d_t[j];
                    weights[i * grad_length + j] -= learn_rate * d_L_d_w[i * grad_length + j];
                    //dest[i * grad_length + j] = d_L_d_w[i * grad_length + j];
                }
                dest[i] = d_L_d_inputs[i];
            }

            free(exp_holder);
            free(d_out_d_t);
            free(d_L_d_t);
            free(d_L_d_w);
            free(d_L_d_inputs);

            return;
        }
};