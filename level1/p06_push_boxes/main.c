#include <stdio.h>
#include <curses.h>

#ifndef MAP_FILES_DIR
#define MAP_FILES_DIR "map files"
#endif

int score=0;
static char  origin=' ';


int movement(char map[][100]) {


    static int  location[2]={1,1};


    for (int i=0;i<15;i++) {
        for (int j=0;j<21;j++) {
            if (map[i][j] == '@') {
                location[0]=i;
                location[1]=j;
            }
        }
    }

    int test1,test2;
    int box1=0,box2=0;
    test1=location[0];
    test2=location[1];

    int no=0;

    int move=getch();

    switch(move) {
        case KEY_UP:  test1-=1  ;   box1=-1;  break;
        case KEY_DOWN:  test1+=1  ; box1=1;   break;
        case KEY_LEFT:  test2-=1  ; box2=-1;  break;
        case KEY_RIGHT: test2+=1  ; box2=1;   break;
    }

    //origin=map[location[0]][location[1]];



    if (map[test1][test2]==' ') {
        switch (origin) {
            case '.': map[location[0]][location[1]]='.';    break;
            case ' ': map[location[0]][location[1]]=' ';    break;
        }
        origin=' ';
        location[0]=test1;
        location[1]=test2;
        map[location[0]][location[1]]='@';
        score++;
    }//player move

    if (map[test1][test2]=='.') {
        switch (origin) {
            case '.': map[location[0]][location[1]]='.';    break;
            case ' ': map[location[0]][location[1]]=' ';    break;
        }
        origin='.';
        location[0]=test1;
        location[1]=test2;
        map[location[0]][location[1]]='@';
        score++;
    }//player move


    if (map[test1][test2] == '$') {
        if (map[test1+box1][test2+box2] != '#'
            && map[test1+box1][test2+box2] == ' ')
        {
            map[location[0]][location[1]]=origin;
            origin=' ';
            location[0]=test1;
            location[1]=test2;
            map[location[0]][location[1]]='@';
            map[test1+box1][test2+box2]='$';
            score++;
        }
        else if (map[test1+box1][test2+box2] == '.') {
            map[location[0]][location[1]]=origin;
            origin=' ';
            location[0]=test1;
            location[1]=test2;
            map[test1+box1][test2+box2]='*';
            map[location[0]][location[1]]='@';
            score++;
        }
    }//player and box move

    if (map[test1][test2] == '*') {
        if (map[test1+box1][test2+box2] != '#'
            && map[test1+box1][test2+box2] == ' ')
        {
            map[location[0]][location[1]]=origin;
            origin='.';
            location[0]=test1;
            location[1]=test2;
            map[location[0]][location[1]]='@';
            map[test1+box1][test2+box2]='$';
            score++;
        }
        else if (map[test1+box1][test2+box2] == '.') {
            map[location[0]][location[1]]=origin;
            origin='.';
            location[0]=test1;
            location[1]=test2;
            map[test1+box1][test2+box2]='*';
            map[location[0]][location[1]]='@';
            score++;
        }
    }//player and box move

    for (int i=0;i<15;i++) {
        for (int j=0;j<21;j++) {
            if (map[i][j] == '$') {
                no++;
                //任务尚未完成
            }
        }
    }
    return no;
}

int main() {

    char maze[100][100];

    int choose_level=0;

    char filename[256];





    FILE *score_fp = fopen(MAP_FILES_DIR "/scores.txt", "a");
    if (score_fp == NULL) {
        perror("打开地图文件失败");
        return 0;
    }


    while (true) {
        score=0;

        origin=' ';

        printf("请输入您想玩的关卡号(输入0退出）：");
        scanf("%d",&choose_level);
        if (choose_level==0) {
            break;
        }
        initscr();              // 进入屏幕模式
        cbreak();               // 不用按回车
        noecho();               // 不显示输入的字符
        keypad(stdscr, TRUE);


        sprintf(filename, MAP_FILES_DIR "/level%d.txt", choose_level);
        FILE *fp = fopen(filename, "r");

        if (fp == NULL) {
            fclose(score_fp);
            endwin();
            perror("打开地图文件失败");
            return 0;
        }
        for (int i=0;i<15;i++) {
            fgets(maze[i], 100, fp);
        }
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
            if (!index) {
                break;
            }
        }
        fprintf(score_fp, "level%d:%d\n",choose_level, score);


        fclose(fp);
    endwin();
    }
    fclose(score_fp);
    return 0;
}
