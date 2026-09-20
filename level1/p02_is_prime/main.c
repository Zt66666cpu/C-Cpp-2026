#include <stdio.h>
#include <stdlib.h>

int is_prime(int n)
{
    if (n==1)
    {
        return 0;
    }
    for (int i = 2; i < n; i++)
    {
        if (n % i == 0)
        {
            return 0;
        }
    }
    return 1;
}
int main() {

    system("chcp 65001 >nul");//解决乱码
    int n=0;
    while (1)
    {
        printf("请输入一个正整数：");
        if ((scanf("%d",&n))!=1)
        {
            printf("您输入的不是数字！\n");
            while (getchar()!='\n');
            continue;
        }
        if (n<=0)
        {
            printf("您输入的不是正整数哦,请重新输入\n\n");
            continue;
        }
        printf("您输入的整数是：%d\n",n);
        if (is_prime(n))
        {
            printf("您输入的是素数\n\n");
        }
        else
        {
            printf("您输入的不是素数\n\n");
        }
    }

    return 0;
}