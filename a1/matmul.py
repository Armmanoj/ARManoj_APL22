import numpy as np

def matmul(A, B):
    # First check if the arguments can be indexed twice
    try:
        for i in A:
            for j in i:
                pass
        for i in B:
            for j in i:
                pass
    except:
        raise TypeError("A and B must be iterable twice")

    # Check if all the rows in each argument are of the same size
    acolno=len(A[0])
    bcolno=len(B[0])
    browno=len(B)
    arowno=len(A)
    for i in A:
        if len(i)!=acolno:
            raise TypeError("All rows of A must have same number of columns")
    for i in B:
        if len(i)!=bcolno:
            raise TypeError("All rows of B must have same number of columns")
    
    # Check if matrices can be multiplied
    if acolno!=browno:
        raise ValueError("A must have as many columns as B has rows")

    # List type matrices are created to handle the different input types which are possible
    
    # Next an O(n^2.807) recursive algorithm is implemented in a different function, to prevent the previous checks from being repeated at each level of the recursion
    # First the matrices have to be padded with 0's to being square with sidelength a power of 2
    N=2**(max(acolno,arowno,bcolno).bit_length())
    AA=[[0]*N]*N
    BB=[[0]*N]*N
    for i in range(arowno):
        temp = list(A[i]) # This is as A or its elements may not be of list type
        temp.extend([0]*(N-acolno))
        AA[i]=temp
    for i in range(browno):
        temp = list(B[i])
        temp.extend([0]*(N-bcolno))
        BB[i]=temp

    def adder(a,b,symbol):
        C=[]
        for i in range(len(a)):
            temp=[]
            for j in range(len(a)):
                temp.append(0)
            C.append(temp)
        for i in range(len(a)):
            for j in range(len(a)):
                try:
                    if symbol=="+":
                        C[i][j]=a[i][j]+b[i][j]
                    else:
                        C[i][j]=a[i][j]-b[i][j]
                except TypeError:
                    raise TypeError("All elements of the matrices must be of numeric type")    
        return C
            
    def Strassens(localA,localB,N):
        if N==2:
            try:
                return [[localA[0][0]*localB[0][0]+localA[0][1]*localB[1][0], 
                         localA[0][0]*localB[0][1]+localA[0][1]*localB[1][1]],
                        [localA[1][0]*localB[0][0]+localA[1][1]*localB[1][0],
                         localA[1][0]*localB[0][1]+localA[1][1]*localB[1][1]]]
            except TypeError:
                raise TypeError("All elements of the matrices must be of numeric type")
        else:
            a=[localA[i][0:N//2] for i in range(N//2)]
            b=[localA[i][N//2:N] for i in range(N//2)]
            c=[localA[i][0:N//2] for i in range(N//2,N)]
            d=[localA[i][N//2:N] for i in range(N//2,N)]
            e=[localB[i][0:N//2] for i in range(N//2)]
            f=[localB[i][N//2:N] for i in range(N//2)]
            g=[localB[i][0:N//2] for i in range(N//2,N)]
            h=[localB[i][N//2:N] for i in range(N//2,N)]
            #The above 8 lines split the 2 matrices into 4 quadrants each
            p1=Strassens(a,adder(f,h,"-"),N//2)
            p2=Strassens(adder(a,b,"+"),h,N//2)
            p3=Strassens(adder(c,d,"+"),e,N//2)
            p4=Strassens(d,adder(g,e,"-"),N//2)
            p5=Strassens(adder(a,d,"+"),adder(e,h,"+"),N//2)
            p6=Strassens(adder(b,d,"-"),adder(g,h,"+"),N//2)
            p7=Strassens(adder(a,c,"-"),adder(e,f,"+"),N//2)
            # Seven matrix multiplications are performed,rather than 8 in a standard multiplication of 2 matrices
            C = []
            for i in range(N):
                temp=[]
                for j in range(N):
                    temp.append(0)
                C.append(temp)
            x=adder(adder(p5,p4,"+"),adder(p2,p6,"-"),"-")
            y=adder(p1,p2,"+")
            z=adder(p3,p4,"+")
            w=adder(adder(p1,p5,"+"),adder(p3,p7,"+"),"-")
            for i in range(0,N//2):
                for j in range(0,N//2):
                    C[i][j]=x[i][j]
            for i in range(0,N//2):
                for j in range(0,N//2):
                    C[i][j+N//2]=y[i][j]
            for i in range(0,N//2):
                for j in range(0,N//2):
                    C[i+N//2][j]=z[i][j]
            for i in range(0,N//2):
                for j in range(0,N//2):
                    C[i+N//2][j+N//2]=w[i][j]
            return C
    Prod=Strassens(AA,BB,N)
    return [Prod[row][:bcolno] for row in range(arowno)]
print(matmul(([1,3,1],[3,4,5],[8,6,7]),[[9,0,1],[1,0,8],[6,5,0]]))
"""C=[[0]*arowno]*bcolno
    element=0
    for i in range arowno:
        for j in range bcolno:
            element=0
            for k in range(acolno):
                try:
                    element+=A[i][k]*B[k][j]
                except:
                    raise TypeError("All elements of the matrix must be of numeric type")
            C[i][j]=element"""
