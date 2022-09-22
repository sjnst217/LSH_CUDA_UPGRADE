
typedef unsigned char UCHAR;
typedef UCHAR* UCHAR_PTR;
typedef unsigned int UINT;
typedef UINT* UINT_PTR;
typedef unsigned long ULONG;
typedef unsigned char BYTE;
typedef unsigned long long ULL;


#define LSH_SUCCESS			0

#define LSH_BLOCK_LEN		128
#define LSH_BLOCK_BIT_LEN	1024
#define LSH_HASH_LEN		32
#define TEST_PT_SIZE		129

typedef struct
{
	BYTE sv_pt[LSH_BLOCK_LEN];
	BYTE sv_last_pt[LSH_BLOCK_LEN];
	UINT padded_pt_len;
	UINT remain_byte_len;
	UINT uChainVar_left[8];		//이렇게 나눈 이유는 이후에 연산을 할 때 편리하게 하기 위해서이다.
	UINT uChainVar_right[8];	//그냥 하나로 써도 상관은 없음
}LSH_Info;

typedef struct
{
	UINT submsg_e_l[8];
	UINT submsg_e_r[8];
	UINT submsg_o_l[8];
	UINT submsg_o_r[8];
}LSH_internal;