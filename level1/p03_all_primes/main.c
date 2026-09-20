#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int is_prime(int n)
{
    if (n==1)
    {
        return 0;
    }
    for (int i = 2; i *i<=n; i++)
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

    clock_t start = clock();

    for (int i = 2; i <= 1000; i++)
    {
        if (is_prime(i))
        {
            printf("%d\n", i);
        }
    }

    clock_t end = clock();
    double seconds = (double)(end - start) / CLOCKS_PER_SEC;
    printf("本次计算耗时：%6f", seconds);

    return 0;
}