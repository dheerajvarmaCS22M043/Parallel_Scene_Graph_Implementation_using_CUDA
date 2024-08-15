
/*
	CS 6023 Assignment 3. 
	Do not make any changes to the boiler plate code or the other files in the folder.
	Use cudaFree to deallocate any memory not in usage.
	Optimize as much as possible.
 */

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>


void readFile (const char *fileName, std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges, std::vector<std::vector<int> > &translations, int &frameSizeX, int &frameSizeY) {
	/* Function for parsing input file*/

	FILE *inputFile = NULL;
	// Read the file for input. 
	if ((inputFile = fopen (fileName, "r")) == NULL) {
		printf ("Failed at opening the file %s\n", fileName) ;
		return ;
	}

	// Input the header information.
	int numMeshes ;
	fscanf (inputFile, "%d", &numMeshes) ;
	fscanf (inputFile, "%d %d", &frameSizeX, &frameSizeY) ;
	

	// Input all meshes and store them inside a vector.
	int meshX, meshY ;
	int globalPositionX, globalPositionY; // top left corner of the matrix.
	int opacity ;
	int* currMesh ;
	for (int i=0; i<numMeshes; i++) {
		fscanf (inputFile, "%d %d", &meshX, &meshY) ;
		fscanf (inputFile, "%d %d", &globalPositionX, &globalPositionY) ;
		fscanf (inputFile, "%d", &opacity) ;
		currMesh = (int*) malloc (sizeof (int) * meshX * meshY) ;
		for (int j=0; j<meshX; j++) {
			for (int k=0; k<meshY; k++) {
				fscanf (inputFile, "%d", &currMesh[j*meshY+k]) ;
			}
		}
		//Create a Scene out of the mesh.
		SceneNode* scene = new SceneNode (i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity) ; 
		scenes.push_back (scene) ;
	}

	// Input all relations and store them in edges.
	int relations;
	fscanf (inputFile, "%d", &relations) ;
	int u, v ; 
	for (int i=0; i<relations; i++) {
		fscanf (inputFile, "%d %d", &u, &v) ;
		edges.push_back ({u,v}) ;
	}

	// Input all translations.
	int numTranslations ;
	fscanf (inputFile, "%d", &numTranslations) ;
	std::vector<int> command (3, 0) ;
	for (int i=0; i<numTranslations; i++) {
		fscanf (inputFile, "%d %d %d", &command[0], &command[1], &command[2]) ;
		translations.push_back (command) ;
	}
}


void writeFile (const char* outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY) {
	/* Function for writing the final png into a file.*/
	FILE *outputFile = NULL; 
	if ((outputFile = fopen (outputFileName, "w")) == NULL) {
		printf ("Failed while opening output file\n") ;
	}
	
	for (int i=0; i<frameSizeX; i++) {
		for (int j=0; j<frameSizeY; j++) {
			fprintf (outputFile, "%d ", hFinalPng[i*frameSizeY+j]) ;
		}
		fprintf (outputFile, "\n") ;
	}
}


__global__ void bfs(int* dOffset, int* dCsr,  int* dVisited, bool* dFinished,  int* dx,  int* dy, int V, int E){
    unsigned  int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < V){
        if(dVisited[id] == 1){ //node is active
            
            for(int nbrId = dOffset[id]; nbrId < dOffset[id+1]; ++nbrId){
                int nbr = dCsr[nbrId];
                if(dVisited[nbr] == 0){
                    dVisited[nbr] = 1;
                    dFinished[0] = false;
                }
                // printf("%d  %d\n", dx[nbr], dy[nbr]);
                // printf("dheeraj\n");
                atomicAdd(&dx[nbr], dx[id]);
                atomicAdd(&dy[nbr], dy[id]);
                // printf("%d %d\n", dx[nbr], dy[nbr]);
            }
            dVisited[id] =  2;
        }
    }
}

__global__ void newXY(int* x1, int* y1, int* x2, int* y2, int* dx, int* dy, int V){
    unsigned  int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < V){
        x2[id] = x1[id] + dx[id];
        y2[id] = y1[id] + dy[id];
    }
}

__global__ void calOpacity(int* ddOpacity, int* dFrameSizeX, int* dFrameSizeY, int V, int* x, int* y, int* dOpacity, int frameSizeX, int frameSizeY){
    unsigned  int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < V){
        for(int len = 0; len < dFrameSizeX[id]; len++){
            for(int wid = 0; wid < dFrameSizeY[id]; wid++){
                int tempX, tempY;
                tempX = x[id] + len;
                tempY = y[id] + wid;
                if(tempX >=0 && tempX < frameSizeX && tempY >= 0 && tempY < frameSizeY)
                 atomicMax(&(ddOpacity[tempX * frameSizeY + tempY]), dOpacity[id]);
            }
        }
    }
}

__global__ void calMeshValue(int* dMesh, int *dMeshOffset, int* ddOpacity, int* dOpacity, int* dFrameSizeX, int* dFrameSizeY, int V, int* x, int* y, int frameSizeX, int frameSizeY, int* dFinalPng){
    unsigned  int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < V){
        for(int len = 0; len < dFrameSizeX[id]; len++){
            for(int wid = 0; wid < dFrameSizeY[id]; wid++){
                int tempX, tempY;
                tempX = x[id] + len;
                tempY = y[id] + wid;
                // printf("Mesh id = %d, x = %d, y = %d, tempX = %d, tempY = %d, dOpacity = %d, cellOpacity = %d, width = %d, height = %d\n", id, x[id], y[id], tempX, tempY, dOpacity[id], ddOpacity[tempX * frameSizeY + tempY], dFrameSizeX[id], dFrameSizeY[id]);
                if(((tempX >=0 && tempX < frameSizeX && tempY >= 0 && tempY < frameSizeY)) && (ddOpacity[tempX * frameSizeY + tempY] == dOpacity[id])) {
                    
                    int meshX = len, meshY = wid;
                        int offset = dMeshOffset[id];
                        dFinalPng[tempX * frameSizeY + tempY] = dMesh[offset + meshX*dFrameSizeY[id] + meshY];
                        
                }
            }
        }
        
    }

}


int main (int argc, char **argv) {
	
	// Read the scenes into memory from File.
	const char *inputFileName = argv[1] ;
	int* hFinalPng ; 

	int frameSizeX, frameSizeY ;
	std::vector<SceneNode*> scenes ;
	std::vector<std::vector<int> > edges ;
	std::vector<std::vector<int> > translations ;
	readFile (inputFileName, scenes, edges, translations, frameSizeX, frameSizeY) ;
	hFinalPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ;
	
	// Make the scene graph from the matrices.
    Renderer* scene = new Renderer(scenes, edges) ;

	// Basic information.
	int V = scenes.size () ;
	int E = edges.size () ;
	int numTranslations = translations.size () ;

	// Convert the scene graph into a csr.
	scene->make_csr () ; // Returns the Compressed Sparse Row representation for the graph.
	int *hOffset = scene->get_h_offset () ;  
	int *hCsr = scene->get_h_csr () ;
	int *hOpacity = scene->get_opacity () ; // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
	int **hMesh = scene->get_mesh_csr () ; // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
	int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX () ; // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
	int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY () ; // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
	int *hFrameSizeX = scene->getFrameSizeX () ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int *hFrameSizeY = scene->getFrameSizeY () ; // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

	auto start = std::chrono::high_resolution_clock::now () ;


	// Code begins here.
	// Do not change anything above this comment.
    
    //step1:
    //dx and dy arrays for translation purpose
     int* x = new  int[V];
     int* y = new  int[V];

    //initializing the dx and dy arrays with 0's 
    for(int i=0;i<V;i++){
        x[i] = 0;
        y[i] = 0;
    }

    //now applying the translations individually:
    for(int i=0;i<numTranslations;i++){
        int n = translations[i][0], c = translations[i][1], a = translations[i][2];
        if(c == 0){
            x[n] -= a;
        }
        else if(c == 1){
            x[n] += a;
        }
        else if(c == 2){
            y[n] -= a;
        }
        else{
            y[n] += a;
        }
        
    }

    // for(int i=0;i<V;i++){
    //     printf("%d  %d\n",x[i], y[i]);
    // }
    // printf("varma\n");


    //step 2:
     int* visited = new  int[V];
    for(int i=0;i<V;i++){
        visited[i] = 0;
    }
    visited[0] = 1;

     int* dVisited; //gpu

    cudaMalloc(&dVisited, sizeof( int)*(V));
    cudaMemcpy(dVisited, visited, sizeof( int)*(V), cudaMemcpyHostToDevice);

    int* dOffset; //gpu
    int* dCsr;
    cudaMalloc(&dOffset, sizeof(int)*(V+1));
    cudaMalloc(&dCsr, sizeof(int)*(E));
    cudaMemcpy(dOffset, hOffset, sizeof(int)*(V+1), cudaMemcpyHostToDevice);
    cudaMemcpy(dCsr, hCsr, sizeof(int)*(E), cudaMemcpyHostToDevice);

     int* dx; //gpu
     int* dy;
    cudaMalloc(&dx, sizeof( int)*(V));
    cudaMalloc(&dy, sizeof( int)*(V));
    cudaMemcpy(dx, x, sizeof(int)*(V), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, sizeof(int)*(V), cudaMemcpyHostToDevice);

    int nBlocks = (V+1023)/(1024);

    while(1){
        bool isFinished[1];
        isFinished[0] = true;
        bool* dFinished; //gpu
        cudaMalloc(&dFinished, sizeof(bool));
        cudaMemcpy(dFinished, isFinished, sizeof(bool), cudaMemcpyHostToDevice);
        
        bfs<<<nBlocks, 1024>>>(dOffset, dCsr, dVisited, dFinished, dx, dy, V, E);
        cudaDeviceSynchronize();

        cudaMemcpy(isFinished, dFinished, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaFree(dFinished);
        if(isFinished[0] == true) break;   
        
    }

    cudaFree(dVisited);
    cudaFree(dCsr);
    cudaFree(dOffset);

    cudaMemcpy(x,dx, sizeof(int)*(V), cudaMemcpyDeviceToHost);
    cudaMemcpy(y,dy, sizeof(int)*(V), cudaMemcpyDeviceToHost);
    // for(int i=0;i<V;i++){
    //     printf("%d  %d\n",x[i], y[i]);
    // }

    //part 3:
     int* x1; //gpu
     int* y1;
    // for(int i=0;i<V;i++){
    //     printf("%d %d\n", hGlobalCoordinatesX[i], hGlobalCoordinatesY[i]);
    // }
    cudaMalloc(&x1, sizeof( int)*(V));
    cudaMalloc(&y1, sizeof( int)*(V));
    cudaMemcpy(x1, hGlobalCoordinatesX, sizeof(int)*(V), cudaMemcpyHostToDevice);
    cudaMemcpy(y1, hGlobalCoordinatesY, sizeof(int)*(V), cudaMemcpyHostToDevice);
     int* x2, *y2; //gpu
    cudaMalloc(&x2, sizeof( int)*(V));
    cudaMalloc(&y2, sizeof( int)*(V));

    newXY<<<nBlocks, 1024>>>(x1, y1, x2, y2, dx, dy, V);  //kernel for calculating new x,y
    cudaDeviceSynchronize();

    // cudaMemcpy(hGlobalCoordinatesX, x2, sizeof( int)*(V), cudaMemcpyHostToDevice);
    // cudaMemcpy(hGlobalCoordinatesY, y2, sizeof( int)*(V), cudaMemcpyHostToDevice);

    cudaFree(x1);
    cudaFree(y1);
    // cudaFree(x2);
    // cudaFree(y2);

    //cuda dx, dy, x2, y2 are alive

    int* ddOpacity;
    cudaMalloc(&ddOpacity, sizeof(int)*(frameSizeX)*(frameSizeY));
    int val = INT_MIN;
    cudaMemset(ddOpacity, val, sizeof(int)*(frameSizeX)*(frameSizeY)); //initialization 

    int* dFrameSizeX, *dFrameSizeY;
    cudaMalloc(&dFrameSizeX, sizeof(int)*(V));
    cudaMalloc(&dFrameSizeY, sizeof(int)*(V));
    cudaMemcpy(dFrameSizeX, hFrameSizeX, sizeof(int)*(V), cudaMemcpyHostToDevice);
    cudaMemcpy(dFrameSizeY, hFrameSizeY, sizeof(int)*(V), cudaMemcpyHostToDevice);

    int* dOpacity;
    cudaMalloc(&dOpacity, sizeof(int)*(V));
    cudaMemcpy(dOpacity, hOpacity, sizeof(int)*(V), cudaMemcpyHostToDevice);

    calOpacity<<<nBlocks, 1024>>>(ddOpacity, dFrameSizeX, dFrameSizeY, V, x2, y2, dOpacity, frameSizeX, frameSizeY); //kernel to calculate max Opacity value of each pixel
    cudaDeviceSynchronize();

    

    //cudaMemcpy(hFinalPng, ddOpacity, sizeof(int)*(frameSizeX)*(frameSizeY), cudaMemcpyDeviceToHost);

    // int** dMesh;
    // cudaMalloc(&dMesh, sizeof(int*)*V);
    // for(int i=0;i<V;i++){
    //     int sizeI = hFrameSizeX[i] * hFrameSizeY[i];
    //     int *dTemp;
    //     cudaMalloc(&dTemp, sizeof(int)*(sizeI));
    //     cudaMemcpy(dTemp, hMesh[i], sizeof(int)*(sizeI), cudaMemcpyHostToDevice);
    //     cudaMemcpy(&dMesh[i] , &dTemp, sizeof(int*) ,cudaMemcpyDeviceToDevice);
    //     cudaFree(dTemp);
        
    // }

    int *dMesh, *dMeshOffset;
    int *hMeshOffset = (int*) malloc (sizeof (int) * (V+1));

    int sum = 0;
    hMeshOffset[0] = 0;
    for(int i=1;i<=V;i++) {
        sum += hFrameSizeX[i-1] * hFrameSizeY[i-1];
        hMeshOffset[i] = sum;
    }
    cudaMalloc(&dMesh, sizeof (int) * sum);
    sum = 0;
    for(int i=0;i<V;i++) {
        
        cudaMemcpy(dMesh + sum, hMesh[i], sizeof(int)*(hFrameSizeX[i] * hFrameSizeY[i]), cudaMemcpyHostToDevice);
        sum += hFrameSizeX[i] * hFrameSizeY[i];
    }
    cudaMalloc(&dMeshOffset, sizeof (int) * (V+1));
    cudaMemcpy(dMeshOffset, hMeshOffset, sizeof(int)*(V+1), cudaMemcpyHostToDevice);

    int* dFinalPng; //gpu
    cudaMalloc(&dFinalPng, sizeof (int) * frameSizeX * frameSizeY);
    cudaMemset(dFinalPng, 0, sizeof(int)*(frameSizeX)*(frameSizeY));

    // for(int i=0;i<V;i++){

    //     int* dMesh;
    //     cudaMalloc(&dMesh, sizeof(int)*(hFrameSizeX[i] * hFrameSizeY[i]));
    //     cudaMemcpy(dMesh, hMesh[i], sizeof(int)*(hFrameSizeX[i] * hFrameSizeY[i]), cudaMemcpyHostToDevice);

        
    // }

    calMeshValue<<<nBlocks, 1024>>>(dMesh, dMeshOffset, ddOpacity, dOpacity, dFrameSizeX, dFrameSizeY, V, x2, y2, frameSizeX, frameSizeY, dFinalPng);
    cudaDeviceSynchronize();

    cudaMemcpy(hFinalPng, dFinalPng, sizeof(int)*(frameSizeX)*(frameSizeY), cudaMemcpyDeviceToHost);

    cudaFree(dMesh);
    cudaFree(ddOpacity);
    cudaFree(dFrameSizeX);
    cudaFree(dFrameSizeY);
    cudaFree(dx);
    cudaFree(dFinalPng);





	// Do not change anything below this comment.
	// Code ends here.

	auto end  = std::chrono::high_resolution_clock::now () ;

	std::chrono::duration<double, std::micro> timeTaken = end-start;

	printf ("execution time : %f\n", timeTaken) ;
	// Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;	

}

