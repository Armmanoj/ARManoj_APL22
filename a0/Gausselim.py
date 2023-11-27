def gausselim(A, B):
    N=len(A)
    for j in range(N):
        # We reduce each column to 0's and 1 in the diagonal position
        if A[j][j]==0 and j!=N-1:
            # Swap rows if there is a 0 on the diagonal
            for k in range(j+1,N):
                if A[k][j]!=0:
                    tempe=A[k]
                    A[k]=A[j]
                    A[j]=tempe
                    tempe=B[k]
                    B[k]=B[j]
                    B[j]=tempe
                else:
                    return -1 # A has 0 determinant
        elif A[j][j]==0 and j==N-1:
            return -1 # A has 0 determinant
        for i in range(N):
            # normalize each row with norm as element of jth column
            norm = A[i][j]
            if norm!=0:
                for k in range(N):
                    A[i][k]/=norm
                B[i]/=norm
        
        for i in range(N): 
            # Substract normalized rows if the value in the jth column is not already 0
            if A[i][j]!=0 and i!=j:
                for k in range(N):
                    A[i][k]-=A[j][k]
                B[i]-=B[j]
    for j in range(N):
        B[j]/=A[j][j]
    return B
print(gausselim([ [2,3], [1,-1] ],[6,0.5]))