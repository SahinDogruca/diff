#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#define DICT_LEN 200
#define TRAIN_LEN 160
#define TEST_LEN  40

FILE * ftext;

void displayHotvectors(bool ** hotVector, int len) {
    int i,j;
    
    for(i = 0; i < TRAIN_LEN; i++) {
        for(j = 0; j < len; j++) {
            printf("%d", hotVector[i][j]);
        }
        printf("\n");
    }
}


void displayDICT(char ** dict, int len) {
    int i;
    for(i = 0; i < len; i++) {
        printf("%d. %s\n", i+1, dict[i]);
    }
}

int contains(char ** dict, int len, char * str) {
    int i;
    for(i = 0; i < len; i++) {
        if(strcmp(dict[i], str) == 0) return i;
    }
    
    return -1;
}

void createDICT(char *** dict_ref, int * dict_len_ref) {
    
    char temp[500];
    char * token;
    
    ftext = fopen("texts.txt", "r");
    
    while(fgets(temp, 500, ftext)) {
       if(!strcmp(temp,"\n")) break;

       token = strtok(temp, " ");

       while(token != NULL) {
           if(contains(*dict_ref, *dict_len_ref, token) == -1) {
               *(dict_ref) = realloc(*dict_ref, sizeof(char*) * ((*dict_len_ref) + 1));
               (*dict_ref)[*dict_len_ref] = (char*) malloc(sizeof(char) * (strlen(token) + 1));
               strcpy((*dict_ref)[*dict_len_ref], token);
               (*dict_ref)[*dict_len_ref][strlen(token)] = '\0';
               (*dict_len_ref)++;

           }

           token = strtok(NULL, " ");
        }

    }
    
    
}

void createHotvectors(bool *** x_ref, char ** dict, int * dict_len_ref) {
    ftext = fopen("texts.txt", "r");
    char * token;
    char temp[500];
    int index, pos = 0;
    
    while(fgets(temp, 500, ftext)) {
       if(!strcmp(temp,"\n")) break;

       token = strtok(temp, " ");

       while(token != NULL) {
           index = contains(dict, *dict_len_ref, token);
           
           if(index != -1) {
               (*x_ref)[pos][index] = true;
           }

           token = strtok(NULL, " ");
        }
        
        pos++;

    }
}

double matrixMul(double * w, bool * x, int len) {
    int i;
    double sum = 0;
    for(i = 0; i < len; i++) {
        sum += (w[i] * x[i]);
    }
    
    return sum;
}

double * gradientDescent(bool ** x, int len, double * w, double eps, int epocs, double pred) {
    double tempW[len], df[len];
    int i, j = 0, k;
    double err = 1.0, cons;

    FILE * fgdloss = fopen("gdLoss.txt", "w");
    
    for(i = 0; i < TRAIN_LEN; i++) {
        cons = 1 - pow(tanh(matrixMul(w, x[i], len)), 2);
        j = 0;
        while(j < epocs && err > pred) {
            
            for(k = 0; k < len; k++) 
                df[k] = cons * x[i][k];
            
            err = 0;
            for(k = 0; k < len; k++) {
                tempW[k] = w[k];
                w[k] -= eps * df[k];
                err += pow((w[k] - tempW[k]), 2);
                
                // printf("i:%d j:%d w_old:%f w_new:%f \n", i,j, tempW[k], w[k]);
            }
            err = sqrt(err);

            char temp[100];
            printf("%.15f\n", err);
            sprintf(temp, "%.15f", err);
            fputs(temp, fgdloss);
            fputs("\n", fgdloss);
            
            j++;
        } 
    }
    
    return w;
}


double * sgd(bool ** x, int len, double * w, double eps, int epocs, double pred) {
    double tempW[len], df[len];
    int i, j = 0, k;
    double err = 1.0, cons;
    FILE * fsgdloss = fopen("fsgdLoss.txt", "w");
    srand(time(NULL));
    int random;
    
    for(i = 0; i < TRAIN_LEN; i++) {
        random = rand() % TRAIN_LEN;
        cons = 1 - pow(tanh(matrixMul(w, x[random], len)), 2);
        j = 0;
        while(j < epocs && err > pred) {
            
            for(k = 0; k < len; k++) {
                df[k] = cons * x[random][k];
            }
            
            err = 0;
            for(k = 0; k < len; k++) {
                tempW[k] = w[k];
                w[k] -= eps * df[k];
                err += pow((w[k] - tempW[k]), 2);
                
                // printf("i:%d j:%d w_old:%f w_new:%f \n", i,j, tempW[k], w[k]);
            }
            
            err = sqrt(err);
            
            j++;
            
        }
        char temp[100];
        printf("%.15f\n", err);
        sprintf(temp, "%.15f", err);
        fputs(temp, fsgdloss);
        fputs("\n", fsgdloss);
    }
    
    return w;
}


double * adam(bool ** x, int len, double * w, double eps, int epocs) {
    const double b1 = 0.9, b2 = 0.999, eps2 = 0.0000000001;
    double m[len+1], v[len+1];
    double mt, vt;
    double gradient[len];
    memset(m, 0, sizeof(double) * len);
    memset(v, 0, sizeof(double) * len);
    
    int i, j, k;
    
    double cons;
    
    // w is the initial guess
    
    for(i = 0; i < TRAIN_LEN; i++) {
        cons = 1 - pow(tanh(matrixMul(w, x[i], len)), 2);
        
        for(j = 1; j < epocs; j++) {
            
            // compute gradient
            for(k = 0; k < len; k++) {
                gradient[k] = cons * x[i][k];
            }
            // update variables
            
            for(k = 0; k < len; k++) {
                m[k] = b1 * m[k] + (1-b1) * gradient[k];
                v[k] = b2 * v[k] + (1-b2) * (gradient[k] * gradient[k]);
                mt = m[k] / (1 - pow(b1, j));
                vt = v[k] / (1 - pow(b2, j));
                
                w[k] = w[k] -  ((eps * mt) / (sqrt(vt) + eps2));
                
                
            }
        }
    }
    
    for(k = 0; k < len; k++) {
        printf("%f\n", w[k]);
    }
    
    return w;
}

void gdTest(bool ** x, int len, double * w, double eps, int epocs, double pred) {
    FILE * fw = fopen("w5.txt", "w");
    int i,j;
    double * weights = gradientDescent(x, len, w, eps, epocs, pred);
    double output = 0;
    
    for(i =TRAIN_LEN ; i < DICT_LEN; i++) {
        output =0;
        for(j = 0; j < len; j++) {
            output += (weights[j] * x[i][j]);
        }
        output = tanh(output);
        
        printf("%d. cümle dogruluk %f\n", i+1 - 160, output);
    }
    
    for(i = 0; i < len; i++) {
        fprintf(fw, "%.5f", w[i]);
        if(i != len-1) {
            fprintf(fw, ",");
        }
        
    }


}

void sgdTest(bool ** x, int len, double * w, double eps, int epocs, double pred) {
    FILE * fw = fopen("w5.txt", "w");
    int i,j;
    double * weights = sgd(x, len, w, eps, epocs, pred);
    
    double output = 0;
    
    for(i =TRAIN_LEN ; i < DICT_LEN; i++) {
        output =0;
        for(j = 0; j < len; j++) {
            output += (weights[j] * x[i][j]);
        }
        output = tanh(output);
        
        printf("%d. cümle dogruluk %f\n", i+1 - 160, output);
    }

    for(i = 0; i < len; i++) {
        fprintf(fw, "%.5f", w[i]);
        if(i != len-1) {
            fprintf(fw, ",");
        }
        
    }
}

void adamTest(bool ** x, int len, double * w, double eps, int epocs){
    FILE * fw = fopen("w5.txt", "w");
    int i,j;
    double * weights = adam(x, len, w, eps, epocs);
    
    double output = 0;
    
    for(i =TRAIN_LEN ; i < DICT_LEN; i++) {
        output =0;
        for(j = 0; j < len; j++) {
            output += (weights[j] * x[i][j]);
        }
        output = tanh(output);
        
        printf("%d. cümle dogruluk %f\n", i+1 - 160, output);
    }

    for(i = 0; i < len; i++) {
        fprintf(fw, "%.5f", w[i]);
        if(i != len-1) {
            fprintf(fw, ",");
        }
        
    }
}


int main() {
    int i;
    char ** dict = (char**) malloc(sizeof(char*) * 0);
    int dict_len = 0;
    
    createDICT(&dict, &dict_len);
    
    // displayDICT(dict, dict_len);
    
    // initialize hot Vectors
    bool ** hotVectors = (bool**) malloc(sizeof(bool*) * DICT_LEN);
    for(i = 0; i < DICT_LEN; i++) {
        hotVectors[i] = (bool*) malloc(sizeof(bool) * dict_len);
        memset(hotVectors[i], 0, sizeof(bool) * dict_len);
    }
    
    createHotvectors(&hotVectors, dict, &dict_len);
    
    // displayHotvectors(hotVectors, dict_len);
    srand(time(NULL));
    double init_values[dict_len];
    for(i = 0; i < dict_len; i++) {
        init_values[i] = (double)rand() / (double)RAND_MAX ;
    }
    
    // adam(hotVectors, dict_len, init_values, 0.001, 1000);

    // gdTest(hotVectors, dict_len, init_values, 0.05, 300, 0.000000001);
    
    sgdTest(hotVectors, dict_len, init_values, 0.05, 300, 0.0000001);
    
   // adamTest(hotVectors, dict_len, init_values, 0.0001, 1000);

    printf("%d", dict_len);
    for(i = 0; i < dict_len; i++) {
        free(dict[i]);
    }
    free(dict);
    
    for(i = 0; i < DICT_LEN; i++) {
        free(hotVectors[i]);
    }
    free(hotVectors);
}
