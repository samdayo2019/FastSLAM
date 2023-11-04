#include "system.h"
//#include <systemc.h>
#include <string>
#include <fstream>
#include <iomanip>
#include <sys/time.h>
#include <queue>
#include <vector>
#include <cstdlib>
#include <random>
#include <iostream>

//stuff 
using namespace std;

Particle::Particle(int num_landmarks)
{
    uint8_t i;
    uint16_t j;
    w = 1.0/num_landmarks; // each particle get uniform weight
    //each particle starts at point 0,0
    x = 0.0; 
    y = 0.0; 
    yaw = 0.0; 
    lm = new float*[num_landmarks];
    lm_cov = new float*[num_landmarks*LM_SIZE];

    //init lm and lm_cov to be all zeroes
    for (i = 0; i < num_landmarks; i++){
        lm[i] = new float[LM_SIZE];
    }

    for (j = 0; j < num_landmarks*LM_SIZE; j++){
        lm_cov[j] = new float[LM_SIZE];
    }

    for(i = 0; i < num_landmarks; i ++){
        for (j = 0; j < LM_SIZE; j++){
            lm[i][j] = 0.0; 
        }
    }

    for(i = 0; i < num_landmarks*LM_SIZE; i ++){
        for (j = 0; j < LM_SIZE; j++){
            lm_cov[i][j] = 0.0; 
        }
    }

    //cout << "we here" << endl; 

}

Particle* fast_slam1(Particle* particles, float* control, float** z, int num_cols){
    particles = predict_particles(particles, control);

    particles = update_with_observation(particles, z, num_cols);
  
    particles = resampling(particles);
   
    return particles; 
}

// control is a 2d vector (v, w)
Particle* predict_particles(Particle* particles, float* control){
    float* px;
    float state[STATE_SIZE];
    px = state;
    float noise[2], prod[2]; //we store noise as a column vector
    float r_mat[2][2]; 
    random_device rd; 
    mt19937 gen(rd()); 
    uniform_real_distribution<float> distribution(0, 1.0); 
    uint8_t i, j, k;

    for (i = 0; i < NUM_PARTICLES; i ++){ 
        // init coln with zeroes
        for (j = 0; j < STATE_SIZE; j++){
            px[j] = 0;
        }

        // load the particle pose into our state vector
        px[0] = particles[i].x;
        px[1] = particles[i].y;
        px[2] = particles[i].yaw;

        // generates noise value for the control value 
        for (k = 0; k < 2; k++){
            noise[k] = distribution(gen); 
        }

        for (j = 0; j < 2; j ++){
            for (k = 0; k < 2; k++){
                r_mat[j][k] = (float)pow(R_matrix[j][k], 0.5);
            }
        }

        for (j = 0; j < 2; j++){
            float sum = 0.0;
            for (k = 0; k < 2; k++){
                sum += noise[k] * r_mat[k][j];
            }
            //prod is kept as a column vector --> conduct a transpose on the noise, r matrix product
            prod[j] = sum;
        }

        //we keep control as a column vector, but it should be a row vector
        prod[0] += control[0];
        prod[1] += control[1];

        px = motion_model(px, prod);
        // cout << px[0] << endl;
        // cout << px[1] << endl;
        // cout << px[2] << endl;
        
        particles[i].x = px[0];
        particles[i].y = px[1];
        particles[i].yaw = px[2];

    }
    return particles; 
}

Particle* update_with_observation(Particle* particles, float** z, int num_cols){     
    uint8_t landmark_id = 0;
    uint8_t i;
    uint8_t j;
    float* obs = new float[3];
    float weight;
    for (i = 0; i < num_cols; i++){
        //cout << (int)i << endl; 
        landmark_id = (uint8_t)z[2][i];
        //cout << "Landmark ID" << (int)landmark_id << endl; 
        obs[0] = z[0][i]; 
        obs[1] = z[1][i];
        obs[2] = z[2][i];        
        for (j = 0; j < NUM_PARTICLES; j++){
            if (abs(particles[j].lm[landmark_id][0]) <= 0.01){
                particles[j] = add_new_landmark(particles[j], obs, Q_matrix);  
            }
            else{             
                weight = compute_weight(particles[j], obs, Q_matrix);               
                particles[j].w *= weight; 
                //cout << "Weight: " << weight << endl;
                particles[j] = update_landmark(particles[j], obs, Q_matrix);
            }
        }
    }

    return particles;
}   

Particle update_landmark(Particle particle, float *z, float (&Q_mat)[2][2]){
    uint8_t lm_id = (uint8_t)z[2];
    float xf[2] = {0.0, 0.0}; // column vector
    float pf[2][2] =  {{0,0}, {0,0}};
    float Hv[2][3] =  {{0, 0, 0}, {0, 0, 0}};
    float Hf[2][2] =  {{0,0}, {0,0}};
    float Sf[2][2] =  {{0,0}, {0,0}};
    float zp[2]; //treat zp as a column vector
    float dz[2]; //treat dx as a column vector

    xf[0] = particle.lm[lm_id][0];
    xf[1] = particle.lm[lm_id][1];

    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            pf[i][j] = particle.lm_cov[2*lm_id + i][j];
        }
    }
    
    compute_jacobians(particle, xf, pf, Q_mat, Hf, Hv, Sf, zp);
    dz[0] = z[0] - zp[0];
    dz[1] = pi_2_pi(z[1] - zp[1]); 

    update_kf_with_cholesky(xf, pf, dz, Q_matrix,  Hf); 

    particle.lm[lm_id][0] = xf[0];
    particle.lm[lm_id][1] = xf[1];

    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            particle.lm_cov[2*lm_id + i][j] = pf[i][j];
        }
    }

    return particle;
}

void update_kf_with_cholesky(float (&xf)[2], float (&pf)[2][2], float (&dz)[2], float (&Q_mat)[2][2], float (&Hf)[2][2]){
    float HfT[2][2]; 
    float PHt[2][2]; 
    float S[2][2]; 
    float ST[2][2];
    float L_matrix[2][2];
    float L_mattrans[2][2];
    float x[2] = {0,0};

    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            HfT[i][j] = Hf[i][j]; 
            PHt[i][j] = pf[i][j];
            ST[i][j] = 0.0;
        }
    }

    transpose_mat(HfT);
    mult_mat(pf, HfT);
    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            PHt[i][j] = HfT[i][j];
        }
    }
    
    mult_mat(Hf, HfT);
    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            ST[i][j] = HfT[i][j] + Q_mat[i][j];
            S[i][j] = ST[i][j];
        }
    }

    transpose_mat(ST);

    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            S[i][j] += ST[i][j];
            S[i][j] *= 0.5;
        }
    }

    cholesky_decomp(S, L_matrix);

    transpose_mat(L_matrix);

    invert_mat(L_matrix);

    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            L_mattrans[i][j] = L_matrix[i][j];
        }
    }

    transpose_mat(L_mattrans);

    mult_mat(PHt, L_matrix); //W1
    mult_mat(L_matrix, L_mattrans); //W

    // x = W*V
    matrix_vector(L_mattrans, dz, x);

    // copy W1 into L_mattrans
    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            L_mattrans[i][j] = L_matrix[i][j];
        }
    }

    transpose_mat(L_mattrans); //W1.T

    mult_mat(L_matrix, L_mattrans); // W1 * W1.T

    // x = W*v + Xf
    xf[0] += x[0]; 
    xf[1] += x[1]; 

    // P = pf - W1*W1.T
    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            pf[i][j] -= L_mattrans[i][j];
        }
    }
}

bool cholesky_decomp(float (&matrix)[2][2], float (&L_matrix)[2][2]){
    L_matrix[0][0] = (float)sqrt(matrix[0][0]); 
    L_matrix[0][1] = 0.0; 
    L_matrix[1][0] = matrix[1][0]/L_matrix[0][0];
    L_matrix[1][1] = (float)sqrt(matrix[1][1] - pow(L_matrix[1][0], 2));

    return true; 


}

Particle add_new_landmark(Particle particle, float *z, float (&Q_mat)[2][2]){
    float r = z[0]; 
    float b = z[1]; 
    uint8_t lm_id = (uint8_t)z[2]; 
    //cout << "Landmark: " << (int) lm_id << endl; 
    float s = sin(pi_2_pi(particle.yaw + b));
    float c = cos(pi_2_pi(particle.yaw + b));

    particle.lm[lm_id][0] = particle.x + r*c; 
    particle.lm[lm_id][1] = particle.y + r*s; 

    // adding the landmark covariance

    float dx = r * c; 
    float dy = r * s; 
    float d2 = (float)(pow(dx, 2) + pow(dy, 2));
    float d = (float)(sqrt(d2));
    float gz[2][2] = {{dx / d, dy / d}, {-dy / d2, dx / d2}};
    float gv_inv[2][2];  
    float gv_trans[2][2];

    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            gv_inv[i][j] = gz[i][j]; 
            gv_trans[i][j] = gz[i][j];
        }
    }
    
    invert_mat(gv_inv);
    transpose_mat(gv_trans); 
    invert_mat(gv_trans);
    mult_mat(Q_mat, gv_trans);
    mult_mat(gv_inv, gv_trans);
    // add the new coveriance matrix to the appropriate landmark position
    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            float val = gv_trans[i][j];
            particle.lm_cov[2*lm_id + i][j] = gv_trans[i][j];
        }
    }
    return particle; 
}

Particle* resampling(Particle* particles){
    float pw[NUM_PARTICLES];
    float n_eff = 0.0;
    float* w_cumulative = new float[NUM_PARTICLES];
    float* base = new float[NUM_PARTICLES];
    float* resample_id= new float[NUM_PARTICLES];
    random_device rd; 
    mt19937 gen(rd()); 
    uniform_real_distribution<float> distribution(0, 1.0); 
    uint8_t indices[NUM_PARTICLES]; 
    Particle tmp_particles[NUM_PARTICLES];

    particles = normalize_weight(particles);
 
    uint16_t i = 0;
    for (i = 0; i < NUM_PARTICLES; i++){
        pw[i] = particles[i].w;
    }

    // for (i = 0; i < NUM_PARTICLES; i++){
    //     n_eff += pow(pw[i], 2);
    // }
    float dp = vector_vector(pw, pw, NUM_PARTICLES);
    n_eff = 1.0/dp;

    if (n_eff < N_RESAMPLE)
    {
        cumulative_sum(pw, w_cumulative);
        
        for (i = 0; i < NUM_PARTICLES; i++){
            pw[i] = (float)1.0/NUM_PARTICLES;
        }
        uint8_t index = 0;
        cumulative_sum(pw, base);
        for (i = 0; i < NUM_PARTICLES; i++){
            base[i] -= 1/NUM_PARTICLES;
            resample_id[i] = distribution(gen);
            resample_id[i] /= NUM_PARTICLES;
            resample_id[i] += base[i];

            while (index < NUM_PARTICLES-1 && resample_id[i] > w_cumulative[index]){
                index++;
            }
            indices[i] = index; 
        }

        for (i = 0; i < NUM_PARTICLES; i++)
        {
           tmp_particles[i] = particles[i];
        }

        uint8_t j, k;

        for(i = 0; i < NUM_PARTICLES; i++){
            particles[i].x = tmp_particles[indices[i]].x;
            particles[i].y = tmp_particles[indices[i]].y;
            particles[i].yaw = tmp_particles[indices[i]].yaw;
            for(j = 0; j < num_landmarks; j++){
                for (k = 0; k < LM_SIZE; k++){
                    particles[i].lm[j][k] = tmp_particles[indices[i]].lm[j][k]; 
                }
            }    

            for(j = 0; j < num_landmarks*LM_SIZE; j++){
                for (k = 0; k < LM_SIZE; k++){
                    particles[i].lm_cov[j][k] = tmp_particles[indices[i]].lm_cov[j][k]; 
                }
            } 
            particles[i].w = 1.0/NUM_PARTICLES; 
        }
    }
    return particles;
}

void cumulative_sum(float* array, float* sum){
    sum[0] = array[0];

    for (uint16_t i = 1; i < NUM_PARTICLES; i++){
        sum[i] = sum[i-1] + array[i];
    }

}

float vector_vector(float* row_vec, float* col_vec, int num_rows){
    float sum = 0.0;
    for (int i = 0; i < num_rows; i++){
        sum += row_vec[i]*col_vec[i];
    }
    return sum; 
}

Particle*   normalize_weight(Particle* particles){
    float sum_weights = 0.0; 
    for (uint16_t i = 0; i < NUM_PARTICLES; i++){
        sum_weights+=particles[i].w;
    }

    if (sum_weights != 0.0){
        for (uint16_t i = 0; i < NUM_PARTICLES; i++){
            particles[i].w /= sum_weights;
        }
    }
    else{
        for (uint16_t i = 0; i < NUM_PARTICLES; i++){
            particles[i].w = 1.0/NUM_PARTICLES;
        }        
    }

    return particles;
}

float compute_weight(Particle particle, float* z, float (&Q_mat)[2][2]){
    uint8_t lm_id = (uint8_t)z[2];
    float xf[2] = {0.0, 0.0}; // column vector
    float pf[2][2] =  {{0,0}, {0,0}};
    float Hv[2][3] =  {{0, 0, 0}, {0, 0, 0}};
    float Hf[2][2] =  {{0,0}, {0,0}};
    float Sf[2][2] =  {{0,0}, {0,0}};
    float zp[2]; //treat zp as a column vector
    float dx[2]; //treat dx as a column vector
    float sInv[2][2] = {{0,0}, {0,0}};    
    float result[2] = {0.0 ,0.0};
    float dotproduct = 0.0; 
    float num = 0.0; 
    float den = 0.0; 
    float w = 0.0;

    xf[0] = particle.lm[lm_id][0];
    xf[1] = particle.lm[lm_id][1];

    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            pf[i][j] = particle.lm_cov[2*lm_id + i][j];
        }
    }
    
    compute_jacobians(particle, xf, pf, Q_mat, Hf, Hv, Sf, zp);
    dx[0] = z[0] - zp[0];
    dx[1] = pi_2_pi(z[1] - zp[1]); 

    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            sInv[i][j] = Sf[i][j];
        }
    }   

    if(!invert_mat(sInv)){
        cout << "Singular" << endl; 
        return 1.0; 
    }
    
    matrix_vector(sInv, dx, result);
    dotproduct = dot_product(dx, result);
    num = (float)exp(-0.5 * dotproduct);
    den = (float)(2.0 * M_PI * sqrt(det(Sf)));
    
    return (num/den);
}

void matrix_vector(float (&matrix)[2][2], float (&vector)[2], float (&result)[2]){
    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

float dot_product(float (&row_vec)[2], float (&col_vec)[2]){
    float sum = 0.0; 
    for (uint8_t i = 0; i < 2; i++){
        sum += row_vec[i]*col_vec[i];
    }
    return sum; 
}

float det(float (&matrix)[2][2]){
    return (matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]);
}

void compute_jacobians(Particle particle, float (&xf)[2], float (&pf)[2][2], float (&Q_mat)[2][2], float (Hf)[2][2], float (&Hv)[2][3], float (&Sf)[2][2], float (&zp)[2]){
    float dx = xf[0] - particle.x; 
    float dy = xf[1] - particle.y;
    float d2 = (float)(pow(dx, 2) + pow(dy, 2));
    float d = (float)sqrt(d2);
    zp[0] = d;
    zp[1] = pi_2_pi(atan2(dy, dx) - particle.yaw);

    float mult_vals[2][2] = {{0, 0}, {0, 0}};
    float vals[2][2] = {{0, 0}, {0, 0}};
    float pf_mat[2][2] = {{0, 0}, {0, 0}};

    Hv[0][0] = -dx/d; 
    Hv[0][1] = -dy/d; 
    Hv[0][2] = 0.0;
    Hv[1][0] = dy/d2; 
    Hv[1][1] = -dx/d2;
    Hv[1][2] = -1.0; 

    Hf[0][0] = dx/d; 
    Hf[0][1] = dy/d;
    Hf[1][0] = -dy/d2; 
    Hf[1][1] = dx/d2; 

    // shallow copy Hv into mult_vals
    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            mult_vals[i][j] = Hf[i][j];
            vals[i][j] = Hf[i][j];
            pf_mat[i][j] = pf[i][j];
        }
    }

    transpose_mat(mult_vals);
    mult_mat(pf_mat, mult_vals);
    mult_mat(vals, mult_vals);
    
    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            Sf[i][j] = mult_vals[i][j] + Q_mat[i][j];
        }
    }
}

void mult_mat(float (&matrix1)[2][2], float (&matrix2)[2][2]){
    float inter[2][2]; 
    uint8_t i = 0;
    uint8_t j = 0;
    uint8_t k = 0;
    for (i = 0; i < 2; i++){
        for (j = 0; j < 2; j++){
            inter[i][j] = matrix2[i][j];
        }
    }

    for (i = 0; i < 2; i++){
        for (j = 0; j < 2; j++){
            matrix2[i][j] = 0;
            for (k = 0; k < 2; k++){
                matrix2[i][j] += matrix1[i][k]*inter[k][j];
            }
        }
    }
}

bool invert_mat(float (&matrix)[2][2]){
    //cout << "We inside the invert" << endl;
    float a = matrix[0][0]; 
    float d = matrix[1][1];
    float b = matrix[0][1];
    float c = matrix[1][0];
    float det = a*d - b*c; 
    if (det == 0.0) return false; 
    matrix[0][0] = d / det; 
    matrix[0][1] = -b / det; 
    matrix[1][0] = -c / det; 
    matrix[1][1] = a / det; 
    return true;
    //cout << "We bout to return from invert" << endl;
}

void transpose_mat(float (&matrix)[2][2]){
    float b = matrix[0][1];
    float c = matrix[1][0];
    matrix[0][1] = c; 
    matrix[1][0] = b;  
}

float* motion_model(float* states, float* control){
    float f[3][3] = {{1.0, 0, 0}, {0, 1.0, 0}, {0, 0, 1.0}};
    float B[3][2] = {{TICK*cos(states[2]), 0}, {TICK*sin(states[2]), 0}, {0.0, TICK}};
    float int_1[3] = {0.0, 0.0, 0.0};
    float int_2[3] = {0.0, 0.0, 0.0};
    float sum = 0.0;

    // compute F*X
    for (int i = 0; i < 3; i++){
        sum = 0.0; 
        for (int j = 0; j < 3; j++){
            sum+= f[i][j]*states[j];
        }
        int_1[i] = sum;
    }
    // compute B*U
    for (int i = 0; i < 3; i++){
        sum = 0.0; 
        for (int j = 0; j < 2; j++){
            sum+= B[i][j]*control[j];
        }
        int_2[i] = sum;
    }

    for (int i = 0; i < 3; i ++){
        states[i] = int_1[i] + int_2[i];
    }

    states[2] = pi_2_pi(states[2]); 
    return states;
}

float pi_2_pi(float value){ 
    return fmod(value + M_PI, 2*M_PI) - M_PI;
}

void calc_input(float time, float* u){
    if(time <= 3.0){
        u[0] = 0.0; 
        u[1] = 0.0;
    }
    else{
        u[0] = 1.0; 
        u[1] = 0.1; 
    }
}

void calc_final_state(Particle* particles, float* xEst){
    xEst[0] = 0;
    xEst[1] = 0; 
    xEst[2] = 0; 

    particles = normalize_weight(particles);

    uint16_t i; 

    for(i = 0; i  < NUM_PARTICLES; i++){
        xEst[0] += particles[i].w*particles[i].x; 
        xEst[1] += particles[i].w*particles[i].y; 
        xEst[2] += particles[i].w*particles[i].yaw; 
    }
    xEst[2] = pi_2_pi(xEst[2]);
}

float** observation(float* xTrue, float* xd, float* u, float** rfid, uint8_t num_id, float* ud, int& num_cols){
    random_device rd; 
    mt19937 gen(rd()); 
    uniform_real_distribution<float> distribution(0, 1.0); 

    xTrue = motion_model(xTrue, u);
    vector<vector<float>> z_new = {{0}, {0}, {0}}; 
    float* zi = new float[3];
    float dx, dy, d, angle, dn, angle_noisy;
    float ** z = new float*[3]; 
    uint8_t i, j; 
    int position = 0; 

    for(i = 0; i < num_id; i++){
        dx = rfid[i][0] - xTrue[0];
        dy = rfid[i][1] - xTrue[1];
        d = (float)hypot(dx, dy); 
        float val = atan2(dy, dx) - xTrue[2];
        angle = (float)pi_2_pi(val);
        if (d <= MAX_RANGE){
            dn = d + distribution(gen)*pow(Q_sim[0][0], 0.5);
            angle_noisy = angle + distribution(gen)*pow(Q_sim[1][1], 0.5);
            zi[0] = dn; 
            zi[1] = pi_2_pi(angle_noisy);
            zi[2] = i;
            if (position == 0){
                z_new[0][0] = zi[0];
                z_new[1][0] = zi[1];    
                z_new[2][0] = zi[2];
            }
            else{
                for(j = 0; j < 3; j++){
                    z_new[j].insert(z_new[j].begin() + position, zi[j]);
                }
            }
            position++;
        }
    }
    num_cols = z_new[0].size();
    // cout << num_cols << endl; 
    for (i = 0; i < 3; i++){
        z[i] = new float[num_cols];
    }

    for(i = 0; i < 3; i++){
        for(j = 0; j < num_cols; j++){
            z[i][j] = z_new[i][j];
        }
    }

    ud[0] = u[0] + distribution(gen)*pow(R_sim[0][0], 0.5);
    ud[1] = u[1] + distribution(gen)*pow(R_sim[1][1], 0.5) + OFFSET_YAW_RATE_NOISE; 

    xd = motion_model(xd, ud);

    return z; 
}

int main()
{
    cout << "We starting FastSLAM execution now!" << endl; 

    float** RFID = new float*[8]; 
    for (int i = 0; i < 8; i++){
        RFID[i] = new float[2]; 
    }
    
    RFID[0][0] = 10.0; 
    RFID[0][1] = -2.0; 
    RFID[1][0] = 15.0; 
    RFID[1][1] = 10.0; 
    RFID[2][0] = 15.0; 
    RFID[2][1] = 15.0; 
    RFID[3][0] = 10.0; 
    RFID[3][1] = 20.0; 
    RFID[4][0] = 3.0; 
    RFID[4][1] = 15.0; 
    RFID[5][0] = -5.0; 
    RFID[5][1] = 20.0; 
    RFID[6][0] = -5.0; 
    RFID[6][1] = 5.0; 
    RFID[7][0] = -10.0; 
    RFID[7][1] = 15.0; 

    num_landmarks = 8; 

    float* xEst = new float[3];
    float* xTrue = new float[3];
    float* xDR = new float[3];

    float* hxEst = xEst; 
    float* hxTrue = xTrue; 
    float* hxDR = xDR; 
    float time = 0;
    float* u = new float(2); 
    float* ud = new float(2); 
    float** z; 
    int num_columns; 


    Particle* particles = new Particle[NUM_PARTICLES]; // working right
    for (int i = 0; i < NUM_PARTICLES; i++){
        particles[i] = Particle(num_landmarks);
    }
    while(150>= time){
        //cout << "=============================================>>>>>>>>>>>>>>>>>>>>>>" << endl;             
        time += TICK; 
        
        // cout << "TIME: " << time << endl;
        calc_input(time, u);
        // cout << "U vector" << endl;
        // cout << u[0] << endl;
        // cout << u[1] << endl; 
        z = observation(xTrue, xDR, u, RFID, num_landmarks, ud, num_columns); 

        particles = fast_slam1(particles, ud, z, num_columns); 

        calc_final_state(particles, xEst);      

        // cout << "Particle Landmarks: " << endl; 
        // for (int i = 0; i < NUM_PARTICLES; i++){
        //     cout << "Particle: " << i << endl; 
        //     for (int j = 0; j < num_landmarks; j++){
        //         for (int k = 0; k < 2; k++){
        //             cout << particles[i].lm[j][k] << " "; 
        //         }
        //         cout << endl; 
        //     }
        // }
        cout << time << endl; 
    }
    cout << "made that shit" << endl; 
    return 1; 
}
