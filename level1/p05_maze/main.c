#include <stdio.h>
#include <curses.h>



int movement(char map[][100]) {

    static int  location[2]={1,1};
    int test1,test2;
    test1=location[0];
    test2=location[1];

    int move=getch();

    switch(move) {
        case KEY_UP:  test1-=1  ; break;
        case KEY_DOWN:  test1+=1  ; break;
        case KEY_LEFT:  test2-=1  ; break;
        case KEY_RIGHT: test2+=1  ; break;
    }


    if (map[test1][test2] == 'E') {
        printw("mission accomplished");
        refresh();
        getch();
        return 1;
    }

    if (map[test1][test2] != '#') {
        map[location[0]][location[1]]=' ';
        location[0]=test1;
        location[1]=test2;
        map[location[0]][location[1]]='@';
    }
    return 0;

}

int main() {

    initscr();              // 进入屏幕模式
    cbreak();               // 不用按回车
    noecho();               // 不显示输入的字符
    keypad(stdscr, TRUE);

    char maze[100][100]={
        "#####################",
        "#@#     #   #       #",
        "# # ### # # ##### # #",
        "# #   #   # #     # #",
        "# ### ##### # #######",
        "#   # # #   #       #",
        "### # # # ### ##### #",
        "# #   #   #   #   # #",
        "# ##### ##### ### # #",
        "# #   # #   #   # # #",
        "# # # # # # ### # # #",
        "#   # #   #   #   # #",
        "# ### ####### ##### #",
        "#   #              E#",
        "#####################",
    };//迷宫

    //上下左右控制
    while(true) {
        clear();
        int index=0;


        for (int i=0;i<15;i++) {
            for (int j=0;j<21;j++) {
                addch(maze[i][j]);
            }
            addch('\n');
        }

        refresh();

        index=movement(maze);
        if (index) {
            break;
        }
    }

    endwin();
    return 0;
}