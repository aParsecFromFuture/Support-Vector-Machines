/*
 *               LINEAR SUPPORT VECTOR MACHINE WITH SEQUENTIAL MINIMAL OPTIMIZATION
 * 
 *        Reference: 
 *        
 *        https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/tutorials/MIT6_034F10_tutor05.pdf
 *        https://youtu.be/_PwhiWxHK8o?list=LLdFcxv-v02rdemKmBVvuwqw
 *        http://cs229.stanford.edu/materials/smo.pdf
 * 
 *                                    STRUCTURE OF DATASET
 * 
 *        |----------|--------------|--------|-----------|-----------|-----|-----------|
 *        | capacity | sample count | stride | sample(1) | sample(2) | ... | sample(n) |
 *        |----------|--------------|--------|-----------|-----------|-----|-----------|
 *        0          1              2        3           6           9     18          21 
 *
 *
 *                                 STRUCTURE OF KERNEL MATRIX
 * 
 *        |-----------|----------------------------------------------------------------|
 *        | dimension |                               data                             |
 *        |-----------|----------------------------------------------------------------|
 *        0           1                                                                37
 */

#include <stdio.h>
#include <stdlib.h>

// BASIC MACROS - DATASET

#define first_index(d) (3)
#define capacity(d) ((int)(d)[0])
#define sample_count(d) ((int)(d)[1])
#define stride(d) ((int)(d)[2])
#define last_index(d) (first_index(d) + sample_count(d) * stride(d))

// DERIVED MACROS - DATASET

#define feature_dim(d) (stride(d) - 1)
#define offset(d, th) (first_index(d) + stride(d) * (th))
#define get_class(d, th) ((int)(d)[offset(d, th) + feature_dim(d)])
#define get_sample(d, th) (&(d)[offset(d, th)])

// MEMORY MACROS - DATASET

#define chunk_size(d) (sizeof((d)[0]) * stride(d))
#define total_memory(d) (chunk_size(d) * capacity(d))
#define used_memory(d) (chunk_size(d) * sample_count(d))

// BASIC MACROS - KERNEL

#define kernel_dim(k) ((int)(k)[0])
#define kernel_at(k, r, c) ((k)[1 + kernel_dim(k) * r + c])

// RANDOM MACRO

#define get_random(m, x) (((x) + (1 + rand() % ((m) - 1))) % (m))

// MIN, MAX, ABS MACROS

#define min(x, y) (((x) < (y))? (x) : (y)) 
#define max(x, y) (((x) > (y))? (x) : (y)) 
#define ABS(x) (((x) < 0)? (-x) : (x))

// L, H, MU MACROS

#define L(ai, aj, yi, yj, C) (((yi) == (yj))? max(0, (ai) + (aj) - (C)) : max(0, (aj) - (ai)))
#define H(ai, aj, yi, yj, C) (((yi) == (yj))? min(C, (ai) + (aj)) : min(C, C + (aj) - (ai)))
#define MU(k, i, j) (2 * kernel_at(k, i, j) - kernel_at(k, i, i) - kernel_at(k, j, j))

// CLIPPING MACROS

#define clip_1(aj, yj, ei, ej, mu) ((aj) - ((yj) * (ei - ej) / (mu)))
#define clip_2(aj, h, l) ((aj) > (h))? (h) : (((aj) >= (l))? (aj) : (l))

// CALCULATION MACROS

#define calc_ai(ai, yi, yj, old_aj, aj) ((ai) + (yi) * (yj) * ((old_aj) - (aj)))

#define calc_b1(b, ei, ai, old_ai, aj, old_aj, xi, xj, yi, yj, k) ((b) - (ei) - (yi) * ((ai) - (old_ai)) * kernel_at(k, xi, xi) - (yj) * ((aj) - (old_aj)) * kernel_at(k, xi, xj))
#define calc_b2(b, ej, ai, old_ai, aj, old_aj, xi, xj, yi, yj, k) ((b) - (ej) - (yi) * ((ai) - (old_ai)) * kernel_at(k, xi, xj) - (yj) * ((aj) - (old_aj)) * kernel_at(k, xj, xj))
#define calc_b(ai, aj, b1, b2, C) ((0 < (ai) && (ai) < (C))? (b1) : (0 < (aj) && (aj) < (C))? (b2) : (((b1) + (b2)) / 2))

void dataset_verbose(const float *dataset);
void kernel_verbose(const float *kernel);
void init_kernel(float *kernel, const float *dataset);
void init_weight(float *dataset, float *alpha, float *weight);
int predict(float *sample, float *weight, float beta, int feature_count);
float y_pred(float *dataset, float *kernel, float *alpha, float beta, int sample_index);
void SMO(float *dataset, float *kernel, float C, float tol, int max_passes, float *alpha, float *beta);
     
int main(int argc, char **argv)
{
    // Training dataset : sample_count: 6, feature_size : 2, classes=(1, -1) 
    
	float dataset[] = {
    6, 6, 3,
    -1, 0, -1,
    0, 1, -1,
    0, 0, -1,
    2, 0, 1,
    1, 0, 1,
    0, -1, 1
    };
    
    float weight[2] = {0};
    float alpha[6] = {0};
    float beta = 0;
    
    // Kernel is a square matrix (sample_count x sample_count)
    
    float kernel[37] = {0};
    kernel[0] = 6;
    
    // Sequentail Minimal Optimization (SMO) parameters
    
    float C = 100000.0;
    float tolerance = 0.0001;
    float max_iter = 100;
    
    init_kernel(kernel, dataset);
    SMO(dataset, kernel, C, tolerance, max_iter, alpha, &beta);
    init_weight(dataset, alpha, weight);
    
    // Information about dataset and kernel matrix
    
    dataset_verbose(dataset);
    kernel_verbose(kernel);
    
    // Print the hyper plane parameters
    
    printf("weight : %.2f, %.2f\n", weight[0], weight[1]);
    printf("beta : %.2f\n", beta);

    // Lets try a test sample
    
    float sample[] = {7, 5};
    int prediction = predict(sample, weight, beta, 2);
    printf("prediction(%.2f, %.2f) : %d\n", sample[0], sample[1], prediction);

    return 0;
}

void dataset_verbose(const float *dataset)
{
    int i;

    printf("capacity : %d\n", capacity(dataset));
    printf("sample count : %d\n", sample_count(dataset));
    printf("feature_dim : %d\n", feature_dim(dataset));
    printf("chunk_size : %d byte\n", chunk_size(dataset));
    printf("total_memory : %d byte\n", total_memory(dataset));
    printf("used_memory : %d byte\n\n", used_memory(dataset));

    for(i = 0; i < sample_count(dataset); i++){
        float *sample = get_sample(dataset, i);
        printf("sample(%d) : (%.2f, %.2f) : %d\n", i, sample[0], sample[1], get_class(dataset, i)); 
    }
    printf("\n");
}

void kernel_verbose(const float *kernel)
{
    int i, j;
    printf("Kernel size : (%d x %d)\n\n", kernel_dim(kernel), kernel_dim(kernel));
    for(i = 0; i < kernel_dim(kernel); i++){
        for(j = 0; j < kernel_dim(kernel); j++){
            printf("%.2f  ", kernel_at(kernel, i, j));
        }
        printf("\n");
    }
    printf("\n");
}

void init_kernel(float *kernel, const float *dataset)
{
    int i, j, k;
    float dot_product;
    const float *sample_1, *sample_2;
    
    for(i = 0; i < kernel_dim(kernel); i++){
        for(j = 0; j < kernel_dim(kernel); j++){
            sample_1 = get_sample(dataset, i);
            sample_2 = get_sample(dataset, j);
            dot_product = 0;
            for(k = 0; k < feature_dim(dataset); k++){
                dot_product += sample_1[k] * sample_2[k];
            }
            kernel_at(kernel, i, j) = dot_product;
        }
    }
}

void init_weight(float *dataset, float *alpha, float *weight)
{
    int i, j, y;
    float *sample;
    
    for(i = 0; i < sample_count(dataset); i++){
        sample = get_sample(dataset, i);
        y = get_class(dataset, i);
        for(j = 0; j < feature_dim(dataset); j++){
            weight[j] += alpha[i] * y * sample[j];
        }
    }
}

int predict(float *sample, float *weight, float beta, int feature_count)
{
    int i;
    float dot_product = 0;
    
    for(i = 0; i < feature_count; i++){
        dot_product += sample[i] * weight[i];
    }
    
    return (((dot_product + beta) >= 0)? 1 : (-1));
}

float y_pred(float *dataset, float *kernel, float *alpha, float beta, int sample_index)
{
    int i;
    float sum = 0;
    
    for(i = 0; i < sample_count(dataset); i++){
        sum += alpha[i] * get_class(dataset, i) * kernel_at(kernel, i, sample_index);
    }
    
    return sum + beta;
}

void SMO(float *dataset, float *kernel, float C, float tol, int max_passes, float *alpha, float *beta){
    int i, j, yi, yj;
    int sample_size = sample_count(dataset);
    int passes = 0;
    int num_changed_alphas;
    float mu_value, l_value, h_value, b1, b2, err_i, err_j, old_alpha_i, old_alpha_j;
    
    while(passes < max_passes){
        num_changed_alphas = 0;
        
        for(i = 0; i < sample_size; i++){
            
            yi = get_class(dataset, i);
            err_i = y_pred(dataset, kernel, alpha, *beta, i) - yi;
            
            if((yi * err_i < -tol && alpha[i] < C) || (yi * err_i > tol && alpha[i] > 0)){
                
                j = get_random(sample_size, i);
                yj = get_class(dataset, j);
                
                err_j = y_pred(dataset, kernel, alpha, *beta, j) - yj;
                
                old_alpha_i = alpha[i];
                old_alpha_j = alpha[j];
                
                l_value = L(alpha[i], alpha[j], yi, yj, C);
                h_value = H(alpha[i], alpha[j], yi, yj, C);
                
                if(l_value == h_value) continue;
                
                mu_value = MU(kernel, i, j);
                
                if(mu_value >= 0) continue;

                alpha[j] = clip_1(alpha[j], yj, err_i, err_j, mu_value);
                alpha[j] = clip_2(alpha[j], h_value, l_value);
                
                if(ABS(alpha[j] - old_alpha_j) < 0.00001) continue;

                alpha[i] = calc_ai(alpha[i], yi, yj, old_alpha_j, alpha[j]);
                
                b1 = calc_b1(*beta, err_i, alpha[i], old_alpha_i, alpha[j], old_alpha_j, i, j, yi, yj, kernel);
                b2 = calc_b2(*beta, err_j, alpha[i], old_alpha_i, alpha[j], old_alpha_j, i, j, yi, yj, kernel);
                
                *beta = calc_b(alpha[i], alpha[j], b1, b2, C);
                
                num_changed_alphas += 1;
            }
        }
        passes = (num_changed_alphas == 0)? (passes + 1) : 0;
    }
}