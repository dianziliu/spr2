#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<random>
#include<math.h>
#include<time.h>
#include<iostream>

typedef double real;

#define PI 3.14159
#define INIT(x,y)  (real *)calloc(x,y)

// 使用了C++11的正太分布
unsigned send = (unsigned)time(0);
std::default_random_engine generator(send);
// 在一些实现中使用(0，0.01)的正态分布
std::normal_distribution<real> distribution(0.0, 0.1);
#define Get_random() distribution(generator)

int debug;
#define DEDUG_V(x,y) if (debug) {printf("%s value is: %s\n",x,y);}
//int n_users, n_items, dim;

int dim = 10;
int n_users = 6400, n_items = 4000;

real lr = 0.007, rg = 0.05;
int iter = 20;
real a = 0.0;
real d = 1;
real * pu, *qi, *du, *di, *dj, *dk;
real *p, *q, *qj, *qk;

char train_path[128];
char u_path[128];	
char i_path[128];	

int analy_args(char * arg, int argc, char ** argv) {
	for (int i = 1; i < argc; i++) {
		if (strcmp(arg, argv[i]) == 0) {
			if (i == argc - 1) {
				exit(-1);
			}
			DEDUG_V(arg, argv[i + 1]);
			//printf("%s value is:%s\n", argv[i], argv[i + 1]);
			return i;
		}
	}
	printf("miss arg of %s\n", arg);
	//exit(-1);
}
int get_args(int argc, char ** argv) {
	int i;

	i = analy_args("-debug", argc, argv);
	debug = atoi(argv[i + 1]);

	i = analy_args("-n_users", argc, argv);
	n_users = atoi(argv[i + 1]);

	i = analy_args("-n_items", argc, argv);
	n_items = atoi(argv[i + 1]);

	i = analy_args("-dim", argc, argv);
	dim = atoi(argv[i + 1]);

	i = analy_args("-iter", argc, argv);
	iter = atoi(argv[i + 1]);
	
	i = analy_args("-lr", argc, argv);
	lr = atof(argv[i + 1]);

	i = analy_args("-rg", argc, argv);
	rg = atof(argv[i + 1]);

	i = analy_args("-train_path", argc, argv);
	strcpy(train_path,argv[i + 1]);

	i = analy_args("-u_path", argc, argv);
	strcpy(u_path, argv[i + 1]);

	i = analy_args("-i_path", argc, argv);
	strcpy(i_path, argv[i + 1]);
	return 0;
}

template<typename A>
A dot(A * a, A * b, int size) {
	A tmp = 0;
	for (int i = 0; i < size; i++) {
		tmp += a[i] * b[i];
	}
	return tmp;
}

real predict(int u, int i) {
	return dot(pu + u*dim, qi + i*dim, dim);
}

real SPR_upgrade(int u, int i, int j, int k) {

	p = pu + u*dim;
	q = qi + i*dim;
	qj = qi + j*dim;
	qk = qi + k*dim;

	real yi = predict(u, i);
	real yj = predict(u, j);
	real yk = predict(u, k);
	real x1 = yi - yj;
	real x2 = yi - yk;
	real z1 = 1 - 1 / (1 + exp(-x1));
	real z2 = 1 - 1 / (1 + exp(-x2));


	for (int i = 0; i < dim; i++) {
		du[i] = lr * (-a * z1 * (q[i] - qj[i]) + d * z2 * (q[i] - qk[i]) -
					  rg * p[i]);
		di[i] = lr * ((-a * z1 + d * z2) * p[i] - rg * q[i]);
		dj[i] = lr * (a * z1 * p[i] - rg * qj[i]);
		dk[i] = lr * (-d * z2 * p[i] - rg * qk[i]);
	}
	for (int i = 0; i < dim; i++) {
		p[i] += du[i];
		q[i] += di[i];
		qj[i] += dj[i];
		qk[i] += dk[i];
	}
	return 0;
}

void train() {

	time_t start = time(NULL);
	int e_count, e_count2;
	char buf[1024];
	int u, i, j, k;
	FILE * fp=fopen(train_path,"r");
	if (fp == nullptr) { printf("open file  %s error\n",train_path); exit(0); }

	for (int local_iter = 0; local_iter < iter; local_iter++) {
		printf("iter is %d|%d\n", local_iter + 1,iter);
		fseek(fp, 0, SEEK_SET);
		e_count = 0;
		e_count2 = 0;
		while (!feof(fp))
		{
			fgets(buf, 1024, fp);
			sscanf(buf, "%d,%d,%d,%d", &u, &i, &j, &k);
			real yi = predict(u, i);
			real yk = predict(u, k);
			real x = yi - yk;
			if (x < -0.5) { e_count++; }
			if (x > 0.5) { e_count2++; }
			SPR_upgrade(u, i, j, k);
		}		
	}
	printf("train end! use %d clock\n", time(NULL) - start);
	fclose(fp);
}

void cheak_args(void) {
	printf("n_users:%d\tn_items:%d\tdim:%d\titer:%d\tlr:%lf\trg:%lf", n_users, n_items, dim, iter, lr, rg);
}

int main(int argc, char ** argv) {
	
	get_args(argc, argv);
	
	pu = INIT(n_users *dim, sizeof(real));
	qi = INIT(n_items *dim, sizeof(real));
	
	for (int i = 0; i < n_users *dim; i++) {
		pu[i] = Get_random();
	}
	for (int i = 0; i < n_items *dim; i++) {
		qi[i] = Get_random();
	}

	du = INIT(dim, sizeof(real));
	di = INIT(dim, sizeof(real));
	dj = INIT(dim, sizeof(real));
	dk = INIT(dim, sizeof(real));

	train();

    // 存储训练结果
	FILE *fp=fopen(u_path,"wb");

	if (fp != nullptr) {
		fwrite(pu, sizeof(real), n_users*dim, fp);
	}
	else
	{
		printf("open u_path error!\n");

	}
	fclose(fp);
	fp=fopen(i_path,"wb");
	
	if (fp != nullptr) {
		fwrite(qi, sizeof(real), n_items*dim, fp);
	}
	else
	{
		printf("open i_path error!\n");

	}
	fclose(fp);
	free(pu);
	free(qi);
	free(du);
	free(di);
	free(dj);
	free(dk);
}