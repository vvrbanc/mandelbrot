#ifndef MANDELMAIN_H
#define MANDELMAIN_H

void mandelbrotCUDA(int *screenbuffer);
void mandelbrotCPU(void *outputBuffer );
void mandelbrotCPUTest(void *outputBuffer );

int renderMandelThread(void *threadData);
int renderMandel2(void *threadData);


#endif