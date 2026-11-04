#include <stdio.h>
#include <stdlib.h>
#include <time.h>


typedef struct Node {
    int data;
    struct Node *next;
} Node;


Node* reverselist(Node *head) {
    Node *prev = NULL;
    Node *curr = head;

    while (curr != NULL) {
        Node *next = curr->next;
        curr -> next = prev;
        prev = curr;
        curr = next;
    }
    return prev;
}
void printlist(Node *head) {
    Node *p = head;
    while (p != NULL) {
        printf("%d\t", p->data);
        p = p->next;
    }
}

void free_list(Node * node);

Node *create_list(int n) {
    Node *head = NULL;
    Node *tail = NULL;

    for (int i=0; i<n; i++) {
       Node * newnode = malloc(sizeof(Node));
        if (newnode == NULL) {
            printf("内存不足\n");
            free_list(head);
            return NULL;
        }
        newnode->data = rand()%10;
        newnode->next = NULL;
        if (head == NULL) {
            head = newnode;
        }
        else {
            tail->next = newnode;
        }
        tail = newnode;
    }

    return head;

}

int find_value_from( Node *head,Node *start_location) {
    Node *p = head;
    int i = 1;
    while (p != NULL && p != start_location) {
        p = p->next;
        i++;
    }
    while (p != NULL) {
        if (p->data == 5) {
            return i;
        }
        p = p->next;
        i++;


    }
    return -1;
}

void free_list(Node *head)
{
    while (head != NULL) {
        Node *next = head->next;  // 先保存下一个节点
        free(head);               // 再释放当前节点
        head = next;              // 移动到下一个节点
    }
}

int main(void) {

    srand((unsigned)time(NULL));
    Node *head = NULL;
    int n;

    printf("请输入您想创建的链表长度：");
    scanf("%d", &n);
    head = create_list(n);

    printf("正向链表为：\n");
    printlist(head);//显示正向链表链表

    printf("\n");

    head = reverselist(head);

    printf("反向链表为：\n");
    printlist(head);//打印反向链表
    printf("\n");

    head = reverselist(head);//还原链表



    int ifkey = 0;
    Node*  location = head;
    int iffind = 0;
    while (ifkey != -1) {
        ifkey = find_value_from(head, location);
        if (ifkey == -1) {
            if (iffind == 0) {
                printf("没找到5\n");
                break;
            }
            else{
                printf("没找到下一个5\n");
                break;
            }
        }
        else {
            printf("在第%d个节点找到了5\n", ifkey);
            location = head;
            for (int i=0; i<ifkey; i++) {
                location = location->next;
            }
            iffind ++;
        }
    }

    free_list(head);
    head = NULL;





    return 0;
}