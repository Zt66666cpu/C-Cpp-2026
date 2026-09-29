#include <stdio.h>
#include <stdlib.h>

void move(char m,char n,char p,int plate_num)
{
    if (plate_num == 1)
    {
        printf("%c -> %c\n",m,n);
    }
    else
    {
        move(m,p,n,plate_num-1);
        printf("%c -> %c\n",m,n);
        move(p,n,m,plate_num-1);
    }
};



int main() {

    system("chcp 65001 >nul");//解决乱码

    int n=0;
    char r='A', s, t;
    printf("移动的圆盘数：");
    scanf("%d", &n);//有几个盘子

    if (n<=0)
    {
        printf("请输入一个正整数！");
        return 0;
    }

    printf("您想从A柱将圆盘移到：");
    scanf(" %c", &s);
    if (s=='B')
    {
        t='C';
    }
    else
    {
        t='B';
    }


    move(r,s,t,n);

    return 0;
}