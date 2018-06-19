#机器学习第二次作业---MDS算法实现
#10142510168 刘恩铭
#  -*-encoding:utf-8-*-
from numpy import *
n=6

# 生成随机的距离的矩阵D,必须是对称的，且对角线元素为0

D=matrix(zeros((n,n)))
for i in range(n):
    for j in range(i+1,n):
        D[i,j]=random.randint(1,100)
D=D+D.T
print('D:',D,'\n')

#计算 dist_i^2,dist_j^2,dist..^2

#dist_i^2,n rows ,1 column
dist_i2=matrix(zeros((n,1)))
for i in range(n):
    for j in range(n):
        dist_i2[i,0]=dist_i2[i,0]+D[i,j]**2
    dist_i2[i,0]=dist_i2[i,0]/n
print("dist_i^2:\n",dist_i2,'\n')

#dist_j^2,1 rows ,n column
dist_j2=matrix(zeros((1,n)))
for j in range(n):
    for i in range(n):
        dist_j2[0,j]=dist_j2[0,j]+D[i,j]**2
    dist_j2[0,j]=dist_j2[0,j]/n

print("dist_j^2:",dist_j2,'\n')

#dist..^2
dist2=0
for i in range(n):
    for j in range(n):
        dist2=dist2+D[i,j]**2
dist2=dist2/(n*n)

print("dist..^2:",dist2,'\n')

# 计算矩阵B
B=matrix(zeros((n,n)))
for i in range(n):
    for j in range(n):
        B[i,j]=-0.5*(D[i,j]**2-dist_i2[i,0]-dist_j2[0,j]+dist2)
print("B:",B,'\n')

# 获取特征值和特征向量

# 计算X,n行d列
a,b=linalg.eig(B)
print('a:',a,'\n')
print('b:',b,'\n')
X = dot(b[:,:2],diag(sqrt(a[:2])))
print("X:",X,'\n')

# 比较原始的数据和低维的距离
print('原始距离','\t新的距离')
for i in range(n):
    for j in range(i+1,n):
        print(D[i,j],'\t\t',str("%.4f"%linalg.norm(X[i]-X[j])))