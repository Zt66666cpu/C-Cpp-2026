//
// Created by strongintheearth on 2026/10/21.
//
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
