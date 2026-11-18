#include <stdint.h>
#include <stdlib.h>
struct ArrowArray { int64_t length, null_count, offset, n_buffers, n_children;
 const void **buffers; struct ArrowArray **children; struct ArrowArray *dictionary;
 void (*release)(struct ArrowArray *); void *private_data; };
struct ArrowSchema { const char *format,*name,*metadata; int64_t flags,n_children;
 struct ArrowSchema **children,*dictionary; void (*release)(struct ArrowSchema *); void *private_data; };
static int releases=0;
static void release_array(struct ArrowArray *a) {
 if (!a || !a->release) return;
 a->release=NULL; free((void*)a->buffers[1]); free((void*)a->buffers); releases++;
}
static void release_schema(struct ArrowSchema *s) { if (s) s->release=NULL; }
int make_array(struct ArrowArray *a,struct ArrowSchema *s,int64_t n) {
 if(n<0 || n>100000) return -1;
 double *v=calloc((size_t)(n?n:1),sizeof(double));
 const void **b=calloc(2,sizeof(void*));
 if(!v || !b) {free(v);free(b);return -1;}
 for(int64_t i=0;i<n;i++) v[i]=(double)i+0.5;
 b[1]=v;
 *a=(struct ArrowArray){n,0,0,2,0,b,NULL,NULL,release_array,NULL};
 *s=(struct ArrowSchema){"g","value",NULL,0,0,NULL,NULL,release_schema,NULL};
 return 0;
}
int release_count(void) { return releases; }
