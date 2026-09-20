#include<stdio.h>
#include<stdlib.h>
#include<windows.h>

#define speed 50

void draw_frame(int index, int scrlen, char letter)
{
    printf("\r");
    for (int i = 1; i <= index; i++)
        printf(" ");
    printf("%c", letter);
    for (int i = 1; i <= scrlen - index - 1; i++)
        printf(" ");
    fflush(stdout);
}

int get_screen_width(void)
{
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info))
        return info.srWindow.Right - info.srWindow.Left + 1;
    return 60;
}

int main(void)
{
    char letter='A';
    int scrlen = get_screen_width();
    int direction=1;
    int index = 0;

    while (1)
    {
        draw_frame(index, scrlen, letter);
        Sleep(speed);
        index+=direction;
        if (index==scrlen-1||index==0)
            direction=-direction;
    }
    return 0;
}
