//************************************************
//*                                               *
//*               Kallisto support                *
//*                                               *
//*************************************************
// функция обратного вызова для отображения информации о ходе вычислений
// pv - лучший вариант
// cm - ход анализируемый в данный момент
typedef void (__stdcall *PF_SearchInfo)(int score, int depth, int speed, char *pv, char *cm);
PF_SearchInfo pfSearchInfo = 0;
// второй вариант
// все параметры строковые
// для улучшенного отображения информации о переборе
typedef void (__stdcall *PF_SearchInfoEx)(char *score, char *depth, char *speed, char **pv, char *cv);
PF_SearchInfoEx pfSearchInfoEx = 0;
#include <fstream>
using namespace std;
   
//*************************************************
//*                                               *
//*               KestoG sources                  *
//*                                               *
//*************************************************

/* includes */
#define WIN32_LEAN_AND_MEAN // exclude rarely used stuff from headers
#include <assert.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>  // for malloc()
#include <string.h>  // for memset()
#include <time.h>
#include <windows.h>

#include "EdAccess.h"
#include "KALLISTO.H" // f-ion prototypes

/* board values */
#define OCCUPIED 0xf0
#define WHITE 1
#define BLACK 2
#define MAN 4
#define KING 8
#define FREE 0
#define MCP_MARGIN 32

#define WHT_MAN 5
#define BLK_MAN 6
#define WHT_KNG 9
#define BLK_KNG  10
/* definitions of new types   */
#define U64 unsigned __int64
#define S64 __int64
#define U32 unsigned __int32
#define  S32 __int32
#define U16 unsigned __int16
#define  S16  __int16
#define U8 unsigned __int8
#define S8  __int8

#define SHIFT_BM 0x180 //  (BLK_MAN << 6) or 384 or 0x180 in hex
#define SHIFT_WM 0x140 // (WHT_MAN << 6) or 320 or 0x140 in hex
#define SHIFT_BK 0x280    // (BLK_KNG << 6) or 640 or 0x280 in hex
#define SHIFT_WK 0x240  //   (WHT_KNG << 6) or 576 or 0x240 in hex

#define MOVE_PIECE( move )       ( (U8)  ((move.m[1]) >> 6 ))
#define CC 3 // change color
#define MAXDEPTH 80
#define MAXMOVES 60
#define MAXCAPTURES 50
#define MAXHIST 524288

#define MATE 30000   // de facto accepted value
#define INFINITY 30001
#define HASHMATE 29900// handle mates up to 100 ply
#define EC_SIZE  ( 0x8000 )
#define EC_MASK ( EC_SIZE - 1 )

#define ED_WIN 30000
#define REHASH 4
#define ETCDEPTH 6  // if depth >= ETCDEPTH do ETC
#define RADIUS 40

//#define KALLISTOAPI WINAPI
//#undef KALLISTO // uncomment if compile to CheckerBoard
#define KALLISTO // uncomment if compile to KallistoGUI

/*----------> compile options  */
// not used options
//#define PERFT
#undef PERFT
#undef MUTE
#undef VERBOSE
#undef STATISTICS

/* getmove return values */
#define DRAW 0
#define WIN 1
#define LOSS 2
#define UNKNOWN 3

#define TICKS CLK_TCK
#define SQUARE_TO_32(square) (SquareTo32[square])

#define EARLY_GAME  ((g_pieces[ (color | KING)  ] > g_pieces[ (color^CC| KING)  ] + 1 ) || ( g_pieces[ (color | KING)  ] + g_pieces[ ( color | MAN ) ] >= 8 ))
#define NOT_ENDGAME ( g_pieces[BLK_KNG] + g_pieces[BLK_MAN] + g_pieces[WHT_KNG ] + g_pieces[WHT_MAN ] > 13 )
#define DO_EXTENSIONS  ( realdepth < depth )

//enum ValueType{
//	NONE=0, UPPER=1, LOWER=2, EXACT= 1 | 2};
const int NONE = 0;
const int UPPER = 1 << 0;
const int LOWER = 1 << 1;
const int EXACT = UPPER | LOWER;
/*	
typedef struct{
     U32  m_lock;
     int  m_value : 32;
     int  m_depth : 8;
     int  m_age : 16;
     U8 m_best_from;
     U8 m_best_to;
     int m_valuetype : 8;
     U8 m_best_from2;
     U8 m_best_to2;
                      }TEntry;
*/
typedef struct{
     U32  m_lock;
     S32  m_value;
     S8  m_depth;
     S16  m_age;
     U8 m_best_from;
     U8 m_best_to;
     S8 m_valuetype;
     U8 m_best_from2;
     U8 m_best_to2;
                      }TEntry;
#define NODE_OPP(type) (-(type))
#define ISWINSCORE(score) ((score) >= HASHMATE)
#define ISLOSSSCORE(score) ((score) <= -HASHMATE)
#define MAX(x, y) (( (x) >= (y)) ? (x) : (y))
#define MIN(x, y) (( (x) <= (y)) ? (x) : (y))
struct coor             /* coordinate structure for board coordinates */
 {
   unsigned int x;
   unsigned int y;
 };

 struct CBmove    // all the information there is about a move
  {
  int jumps;               // how many jumps are there in this move?
  int newpiece;            // what type of piece appears on to
  int oldpiece;            // what disappears on from
  struct  coor from, to;            // coordinates of the piece in 8x8 notation
  struct coor path[12];            // intermediate path coordinates
  struct  coor del[12];            // squares whose pieces are deleted
  int delpiece[12];            // what is on these squares
  } GCBmove;

struct move2
  {
    unsigned short m[12];
    char path[11]; // was short path[12];
    unsigned char l; // move's length
  };
  
 struct move
  {
    unsigned short m[2];
  }; 

/* function prototypes */

static int          compute( int b[46],int color, int time, char output[256]);
static int          PVSearch(int b[46],int depth,int alpha,int beta,int color);
static int          Search(int b[46],int depth,int beta ,int color,int node_type,bool mcp);
static int          LowDepth(int b[46],int depth,int beta ,int color);
static int          rootsearch(int b[46], int alpha, int beta, int depth,int color,int search_type);
static int          QSearch(int b[46], int alpha,int beta,int color);
static void       QuickSort( int SortVals[MAXMOVES],struct move2 movelist[MAXMOVES], int inf, int sup);
static int          evaluation(int b[46], int color, int alpha, int beta);
static int          fast_eval(int b[46],int color);
static void       domove(int *b,struct move2 *move,int stm);
static void       domove2(int *b,struct move2 *move,int stm);
static void __inline  doprom(int *b,struct move *move,int stm );
static void __inline  undoprom(int *b,struct move *move,int stm );
static void       undomove(int *b,struct move2 *move,int stm);
static void       update_hash(struct move2 *move);
static unsigned int       Gen_Captures(int *b,struct move2 *movelist, int color, unsigned start );
static unsigned int       Gen_Moves(int *b,struct move2 *movelist, int color );
static unsigned int       Gen_Proms(int *b,struct move *movelist,int color );
static void                     black_king_capture(int *b,int *n,struct move2 *movelist, int j,int in_dir);
static void                     black_man_capture(int *b,int *n,struct move2 *movelist, int j,int in_dir);
static void                     white_king_capture(int *b,int *n,struct move2 *movelist, int j,int in_dir);
static void                     white_man_capture(int *b,int *n,struct move2 *movelist, int j,int in_dir);
static unsigned int        Test_Capture( int *b, int color);
static int          Test_From_pb( int *b,int mloc,int dir);
static int          Test_From_cb( int *b, int mloc,int dir);
static int          Test_From_pw( int *b,int mloc,int dir);
static int          Test_From_cw( int *b, int mloc,int dir);

static void         setbestmove(struct move2 move);
static void         MoveToStr(move2 m, char *s);
static struct       coor numbertocoor(int n);
static void         movetonotation(struct move2 move,char str[80]);

static U64           rand64(void);
static U64           Position_to_Hashnumber( int b[46] , int color );
static void           Create_HashFunction(void);
static void           TTableInit( unsigned int size);
static void           retrievepv( int b[46], char *pv, int color);
static int hashretrieve(int *tr_depth, int depth,int *value,int alpha,int beta,U8 *best_from,U8 *best_to,U8 *best_from2,U8 *best_to2, int *try_mcp);
static void       hashstore(int value,int depth,U8 best_from,U8 best_to,int f);
static int          value_to_tt( int value );
static void       ClearHistory( void );
static void       Perft(int *b,int color,unsigned depth);
static void __inline    killer( U8 bestfrom,U8 bestto,int realdepth,int capture);
static void                  hist_succ( U8 from,U8 to,int depth,int capture);
static void                  history_bad( U8 from,U8 to,int depth );
static  bool                  is_move_to_7( int b[46],struct move2 *move );
static int __inline       is_promotion(struct move2 *move);
static int                     is_blk_kng( int b[46] );
static int                     is_wht_kng( int b[46] );
static int                     is_blk_kng_1( int b[46] );
static int                      is_wht_kng_1( int b[46] );
static int                      is_blk_kng_2( int b[46] );
static int                      is_wht_kng_2( int b[46] );
static int                      is_blk_kng_3( int b[46] );
static int                      is_wht_kng_3( int b[46] );
static int                      is_blk_kng_4( int b[46] );
static int                      is_wht_kng_4( int b[46] );
static int                      pick_next_move( int *marker,int SortVals[MAXMOVES],int n );
static void                   Sort( int start,int num, int SortVals[MAXMOVES],struct move2 movelist[MAXMOVES] );
static void                   init_piece_lists( int b[46] );
__inline int                 value_mate_in(int ply);
__inline int                 value_mated_in(int ply);
static bool                    move_to_a1h8( struct move2 *move );
static int                      has_man_on_7th(int b[46],int color);
static int                      EdProbe(int c[46],int color);
static __inline             U8  FROM( struct move2 *move );
static __inline             U8  TO( struct move2 *move );
static     void               EvalHashClear(void);

/*----------> globals  */
CRITICAL_SECTION AnalyseSection;
int *play;
static U64 ZobristNumbers[41][16];
static U64 HashSTM; // random number - side to move
static U64 HASH_KEY; // global variable HASH_KEY
static U64 MASK;
unsigned int size = 8; // default size 8 MB
int realdepth;
unsigned int nodes;
struct move2 bestrootmove;
static U64 RepNum[MAXDEPTH];
static int history_tabular[1024]; // array used by history heuristics

// killer slots
static U8 killersf1[MAXDEPTH+2];
static U8 killerst1[MAXDEPTH+2];

static U8 killersf2[MAXDEPTH+2];
static U8 killerst2[MAXDEPTH+2];

double PerftNodes;
static const int directions[4] =  { -0x5,-0x4,0x5,0x4 };
static const int NodeAll = -1;
static const int NodePV  =  0;
static const int NodeCut = +1;
static const int MVALUE[11] = {0,0,0,0,0,100,100,0,0,250,250};
static int g_pieces[11]; // global array
double start,t,maxtime; // time variables
const int SearchNormal = 0;
const int SearchShort  = 1;

const int SquareTo32[41] = { 0,0,0,0,0,           // 0 .. 4
	                                            0,1,2,3,0,           // 5 .. 8 (9)
                                                 4,5,6,7,             // 10 .. 13	                                            
                                                 8,9,10,11,0,       // 14 .. 17 (18)	                                            
                                                 12,13,14,15,       // 19 .. 22
                                                  16,17,18,19,0,    // 23 .. 26 (27)
                                                  20,21,22,23,       // 28 .. 31
                                                  24,25,26,27,0,      // 32 .. 35 (36)
                                                  28,29,30,31          // 37 .. 40                                          
                                                 };

static unsigned int indices[41]; // indexes into p_list for a given square
static unsigned int p_list[3][16]; // p_list contains the location of all the pieces of either color
static unsigned int c_num[MAXDEPTH+1][16]; // captured number

unsigned int num_wpieces;
unsigned int num_bpieces;
int  g_Panic; // add more time
int  g_seldepth; // selective depth i.e.  maximal reached depth
int g_root_mb;    // root material balance
unsigned int searches_performed_in_game = 0; // age or generation
TEntry *ttable = NULL;
EdAccess *ED = NULL;

unsigned int EdPieces = 0;
bool Reversible[MAXDEPTH+1];
int EdRoot[3];
bool EdNocaptures = false;
static bool inSearch;

 const int GrainSize = 2;
 U64 EvalHash[ EC_SIZE ];
 bool ZobristInitialized = false;
//#pragma warn -8057 // disable warning #8057 when compiling
//#pragma warn -8004 // disable warning #8004 when compiling

/*    dll stuff                  */
BOOL WINAPI
DllEntryPoint (HANDLE hDLL, DWORD dwReason, LPVOID lpReserved)
{
  /* in a dll you used to have LibMain instead of WinMain in
     windows programs, or main in normal C programs win32
     replaces LibMain with DllEntryPoint. */

  switch (dwReason)
    {
    case DLL_PROCESS_ATTACH:
      /* dll loaded. put initializations here */
      break;
    case DLL_PROCESS_DETACH:
      /* program is unloading dll. put clean up here */
      break;
    case DLL_THREAD_ATTACH:
      break;
    case DLL_THREAD_DETACH:
      break;
    default:
      break;
    }
  return TRUE;
}


/* CheckerBoard API: enginecommand(), islegal(), getmove() */

int WINAPI
enginecommand (char str[256], char reply[1024])
{
  /* answer to commands sent by CheckerBoard.  This does not
   * answer to some of the commands, eg it has no engine
   * options. */

  char command[256], param1[256], param2[256];
  char *stopstring;
  sscanf (str, "%s %s %s", command, param1, param2);

  // check for command keywords

  if (strcmp (command, "name") == 0)
    {
      sprintf (reply, "KestoG engine 1.5");
      return 1;
    }

  if (strcmp (command, "about") == 0)
    {
     sprintf (reply,"KestoG engine 1.5\n by Kestutis Gasaitis\nEngine which plays Russian checkers\n2005-2011\nE-mail to:kgasaitis@yahoo.com with any comments.");
     return 1;
    }

  if (strcmp (command, "help") == 0)
    {
    //sprintf(reply,"Rules of Play.htm");
    return 0;
    }

  if (strcmp (command, "set") == 0)
    {
      if (strcmp (param1, "hashsize") == 0)
         {
              size = strtol( param2, &stopstring, 10 );
              if ( size < 1) return 0;
              if ( size > 128) size = 128;
              return 1;
          }
     if (strcmp (param1, "book") == 0)
   {
     return 0;
   }
    }

  if (strcmp (command, "get") == 0)
    {
      if (strcmp (param1, "hashsize") == 0)
   {
     return 0;
   }
      if (strcmp (param1, "book") == 0)
   {
     return 0;
   }
      if (strcmp (param1, "protocolversion") == 0)
   {
     sprintf (reply, "2");
     return 1;
   }
      if (strcmp (param1, "gametype") == 0)
   {
     sprintf (reply, "25"); // 25 stands for Russian checkers
     return 1;
   }
    }
    strcpy (reply, "?");
  return 0;
}


int WINAPI
islegal (int b[8][8], int color, int from, int to,struct CBmove *move)
{
  /* islegal tells CheckerBoard if a move the user wants to
   * make is legal or not. to check this, we generate a
   * movelist and compare the moves in the movelist to the
   * move the user wants to make with from & to */

    int n,i,found=0,Lfrom,Lto;
    struct move2 movelist[MAXMOVES];
    int board[46];
    int capture;
    char Lstr[80];

   /* initialize board */
   for(i=45;i>=0;i--)
     board[i]=OCCUPIED;
   for(i=5;i<=40;i++)
     board[i]=FREE;
       board[5]=b[0][0];board[6]=b[2][0];board[7]=b[4][0];board[8]=b[6][0];
       board[10]=b[1][1];board[11]=b[3][1];board[12]=b[5][1];board[13]=b[7][1];
       board[14]=b[0][2];board[15]=b[2][2];board[16]=b[4][2];board[17]=b[6][2];
       board[19]=b[1][3];board[20]=b[3][3];board[21]=b[5][3];board[22]=b[7][3];
       board[23]=b[0][4];board[24]=b[2][4];board[25]=b[4][4];board[26]=b[6][4];
       board[28]=b[1][5];board[29]=b[3][5];board[30]=b[5][5];board[31]=b[7][5];
       board[32]=b[0][6];board[33]=b[2][6];board[34]=b[4][6];board[35]=b[6][6];
       board[37]=b[1][7];board[38]=b[3][7];board[39]=b[5][7];board[40]=b[7][7];
   for(i=5;i<=40;i++)
     if(board[i] == 0) board[i]=FREE;
   for(i=9;i<=36;i+=9)
     board[i]=OCCUPIED;
     init_piece_lists(board);

     /* board initialized */

     n = Gen_Captures( board,movelist,color,1);
     capture=n;

     if (!n)
     n = Gen_Moves( board,movelist,color );
     if (!n) return 0;

     /* now we have a movelist - check if from and to are the same */
    for(i=0;i<n;i++)
      {
                     movetonotation(movelist[i],Lstr);
                     if ( capture )
                        sscanf(Lstr,"%i%*c%i",&Lfrom,&Lto);
                    else
                        sscanf(Lstr,"%i%*c%i",&Lfrom,&Lto);

                    if((from==Lfrom) && (to==Lto))
         {
         found=1;
         break;
         }
      }
      if(found){
      /* sets GCBmove to movelist[i] */
       setbestmove(movelist[i]);
      *move=GCBmove;
        }

    return found;
   }


int WINAPI
getmove (int board[8][8],int color,double maxtime,char str[1024],int *playnow,int info,int unused,struct CBmove *move)
{
  /* getmove is what checkerboard calls. you get the parameters:

     - b[8][8]
     is the current position. the values in the array are
     determined by the #defined values of BLACK, WHITE, KING,
     MAN. a black king for instance is represented by BLACK|KING.

     - color
     is the side to make a move. BLACK or WHITE.

     - maxtime
     is the time your program should use to make a move. this
     is what you specify as level in checkerboard. so if you
     exceed this time it's not too bad - just don't exceed it
     too much...

     - str
     is a pointer to the output string of the checkerboard status bar.
     you can use sprintf(str,"information"); to print any information you
     want into the status bar.

     - *playnow
     is a pointer to the playnow variable of checkerboard. if
     the user would like your engine to play immediately, this
     value is nonzero, else zero. you should respond to a
     nonzero value of *playnow by interrupting your search
     IMMEDIATELY.

     - CBmove
     tells checkerboard what your move is, see above.
   */

   int i;
   int value;
   int b[46];
   int time = (int)(maxtime * 1000.0);
   double t, elapsed;
   /* initialize board */
   for(i=45;i>=0;i--)
     b[i]=OCCUPIED;
   for(i=5;i<=40;i++)
     b[i]=FREE;
          /*    (white)
                37  38  39  40
              32  33  34  35
                28  29  30  31
              23  24  25  26
                19  20  21  22
              14  15  16  17
                10  11  12  13
               5   6   7   8
         (black)   */
     b[5]=board[0][0];b[6]=board[2][0];b[7]=board[4][0];b[8]=board[6][0];
     b[10]=board[1][1];b[11]=board[3][1];b[12]=board[5][1];b[13]=board[7][1];
     b[14]=board[0][2];b[15]=board[2][2];b[16]=board[4][2];b[17]=board[6][2];
     b[19]=board[1][3];b[20]=board[3][3];b[21]=board[5][3];b[22]=board[7][3];
     b[23]=board[0][4];b[24]=board[2][4];b[25]=board[4][4];b[26]=board[6][4];
     b[28]=board[1][5];b[29]=board[3][5];b[30]=board[5][5];b[31]=board[7][5];
     b[32]=board[0][6];b[33]=board[2][6];b[34]=board[4][6];b[35]=board[6][6];
     b[37]=board[1][7];b[38]=board[3][7];b[39]=board[5][7];b[40]=board[7][7];
   for(i=5;i<=40;i++)
       if ( b[i] == 0 ) b[i]=FREE;
   for(i=9;i<=36;i+=9)
       b[i]=OCCUPIED;
       play=playnow;

#ifdef PERFT
       t=clock();
       init_piece_lists(b);
       PerftNodes = 0;
       realdepth = 0;
       Perft(b,color,10);
       elapsed = (clock()-t)/(double)CLK_TCK;
       sprintf(str,"[done][time %.2fs][PerftNodes %.0f]" ,elapsed,PerftNodes);
#else
     // init_piece_lists(b);
      if (( info & 1 ) || ( ttable == NULL)){
      TTableInit(size);
      EvalHashClear();
      searches_performed_in_game = 0;
      Create_HashFunction();
      ClearHistory();
                                                                     }
      value = compute(b, color,time, str);
#endif

   for(i=5;i<=40;i++)
       if( b[i] == FREE) b[i]=0;
     /* return the board */
    board[0][0]=b[5];board[2][0]=b[6];board[4][0]=b[7];board[6][0]=b[8];
    board[1][1]=b[10];board[3][1]=b[11];board[5][1]=b[12];board[7][1]=b[13];
    board[0][2]=b[14];board[2][2]=b[15];board[4][2]=b[16];board[6][2]=b[17];
    board[1][3]=b[19];board[3][3]=b[20];board[5][3]=b[21];board[7][3]=b[22];
    board[0][4]=b[23];board[2][4]=b[24];board[4][4]=b[25];board[6][4]=b[26];
    board[1][5]=b[28];board[3][5]=b[29];board[5][5]=b[30];board[7][5]=b[31];
    board[0][6]=b[32];board[2][6]=b[33];board[4][6]=b[34];board[6][6]=b[35];
    board[1][7]=b[37];board[3][7]=b[38];board[5][7]=b[39];board[7][7]=b[40];

    /* set the move */
   *move=GCBmove;
    if(value>=HASHMATE) return WIN;
    if(value<=-HASHMATE) return LOSS;
  return UNKNOWN;
}

__inline int value_mate_in(int ply){
  return (MATE - ply);
}

__inline int value_mated_in(int ply){
  return (-MATE + ply);
}

static void black_king_capture( int *b,int *n,struct move2 *movelist,int j,int in_dir){
   //
   int capsq,dir,found_cap=0,found_pd,next_dir,temp;
   unsigned i,m;
   struct move2 move,orgmove;
   //orgmove = movelist[*n];
               for ( i =0; i < 4; i++ ){    // scan all 4 directions
               if ( in_dir & (1 << i) == 0 ) continue;
               dir = directions[i];
               temp = j; // from square
               do temp += dir; while ( b[temp] == FREE );
               if ( ( b[temp] & WHITE ) != 0 ){
                 temp = temp + dir;
                 if ( b[temp] ) continue;
                 capsq = temp - dir; // fix captured square address in capsq
                 found_pd = 0;
                 if ( found_cap == 0 ) orgmove = movelist[*n];
                 do{
                      if ( Test_From_pb(b,temp,dir) ){
                            // add to movelist
                            move = orgmove;
                            move.path[move.l - 1] = temp;
                            m = SHIFT_BK|temp;
                            move.m[1] = m;
                            m = (b[capsq]<<6)|capsq;
                            move.m[move.l] = m;
                            move.l++;
                            found_pd++;
                            found_cap++;
                            movelist[*n] = move;
                            b[capsq]++;
     // further jumps
     switch (i){
     case 0:next_dir = ( found_pd == 1 ) ? 13:5;break; // in binary form 1101:0101
     case 1:next_dir = ( found_pd == 1 ) ? 14:10;break; // in binary form 1110:1010
     case 2:next_dir = ( found_pd == 1 ) ? 7:5;break; // in binary form 0111:0101
     case 3:next_dir = ( found_pd == 1 ) ? 11:10;break; // in binary form 1011:1010
               }
                            black_king_capture(b, n, movelist, temp,next_dir);
                            b[capsq]--;
                                                                  } // if
                            temp = temp + dir;
                            } while ( b[temp] == FREE );

                            if ( found_pd == 0 ){
                            if ( (b[temp] & WHITE) != 0 && (b[temp+dir] == FREE)){
                            temp = capsq + dir;
                                  // add to movelist
                                  move = orgmove;
                                  move.path[move.l - 1] = temp;
                                  m = SHIFT_BK|temp;
                                  move.m[1] = m;
                                  m = (b[capsq]<<6)|capsq;
                                  move.m[move.l] = m;
                                  move.l++;
                                  found_cap++;
                                  movelist[*n] = move;
                                  b[capsq]++;
                                  // further jump in same direction
                                  next_dir = 1 << ( 3 - i );
                                  black_king_capture(b, n, movelist, temp,next_dir);
                                  b[capsq]--;
                                } // if
                                  else{
                                  temp = capsq + dir;
                                  do{
                                  // add to movelist
                                  move = orgmove;
                                  move.path[move.l - 1] = temp;
                                  m = SHIFT_BK|temp;
                                  move.m[1] = m;
                                  m = (b[capsq]<<6)|capsq;
                                  move.m[move.l] = m;
                                  move.l++;
                                  found_cap++;
                                  movelist[*n] = move;
                                  (*n)++;
                                  temp = temp + dir;
                                       }while ( b[temp] == FREE );
                                          }
                                       }
                     } //   /*   ----===========----   */
                } // for

     if ( found_cap == 0 ) (*n)++;
 }


static void black_man_capture( int *b,int *n,struct move2 *movelist,int j,int in_dir){
 //
   int dir,found_cap=0,sq1,sq2;
   unsigned i,m;
   struct move2 move,orgmove;
              for ( i = 0; i < 4; i++ ){ // scan all 4 directions
                      dir = directions[i];
                      if ( dir == in_dir ) continue;
                      sq1 = j + dir;
                      if (  ( b[sq1] & WHITE ) != 0 ){
                      sq2 = j + (dir<<1);
                          if ( b[sq2] == FREE ){
                                     // add to movelist
                                     if ( found_cap == 0 ) orgmove = movelist[*n];
                                     move = orgmove;
                                     move.path[move.l - 1] = sq2;
                                     m = (b[sq1]<<6)|sq1;
                                     move.m[move.l] = m;
                                     move.l++;
                                     if ( sq2 > 35 ){  // promotion
                                     m = SHIFT_BK|sq2;
                                     move.m[1] = m;
                                     movelist[*n] = move;
                                     found_cap++;
                                     if ( (sq2 == 37) || (sq2 == 40) ){
                                     (*n)++;
                                     continue;
                                                                                         }
                                     else{
                                     //dir = (dir == 4)?1:2;
                                     dir = dir - 3;
                                     b[sq1]++;
                                     // further jump as king
                                     black_king_capture(b, n, movelist,sq2,dir);
                                     b[sq1]--;
                                            }
                                                           } // promotion
                                     else{     // non-promotion
                                     m = SHIFT_BM|sq2;
                                     move.m[1] = m;
                                     found_cap++;
                                     movelist[*n] = move;
                                     b[sq1]++;
                                     //
                                     black_man_capture(b, n, movelist,sq2,-dir);
                                     b[sq1]--;                                 
                                           }       	
                                     } //
                                  } //
                            } // for
        if ( found_cap == 0 ) (*n)++;
}


static void white_king_capture( int *b,int *n,struct move2 *movelist,int j,int in_dir){
   //
   int capsq,dir,found_cap=0,found_pd,next_dir,temp;
   unsigned i,m;
   struct move2 move,orgmove;
   //orgmove = movelist[*n];
              for ( i = 0; i < 4; i++ ){ // scan all 4 directions
              if ( in_dir & (1 << i) == 0 ) continue;
              dir = directions[i];
              temp = j; // from square
              do temp += dir;while ( b[temp] == FREE );
              if ( ( b[temp] & BLACK ) != 0 ){ 
              temp = temp + dir;
              if ( b[temp] ) continue;
              capsq = temp - dir; // fix captured square address in capsq
              found_pd = 0;
              if ( found_cap == 0 ) orgmove = movelist[*n];
               do{
                    if ( Test_From_pw(b,temp,dir) ){
                     // add to movelist
                     move = orgmove;
                     move.path[move.l - 1] = temp;
                     m = SHIFT_WK|temp;
                     move.m[1] = m;
                     m = (b[capsq]<<6)|capsq;
                     move.m[move.l] = m;
                     move.l++;
                     found_pd++;
                     found_cap++;
                     movelist[*n] = move;
                     b[capsq]--;
                     // further jumps
         switch (i){
         case 0:next_dir = ( found_pd == 1 ) ? 13:5;break; // in binary form 1101:0101
         case 1:next_dir = ( found_pd == 1 ) ? 14:10;break; // in binary form 1110:1010
         case 2:next_dir = ( found_pd == 1 ) ? 7:5;break; // in binary form 0111:0101
         case 3:next_dir = ( found_pd == 1 ) ? 11:10;break; // in binary form 1011:1010
                   }
                     white_king_capture(b, n, movelist, temp,next_dir);
                     b[capsq]++;
                                 } // if
                     temp = temp + dir;
                     } while ( b[temp] == FREE );

                     if ( found_pd == 0 ){
                     if ( (b[temp] & BLACK) != 0 && (b[temp+dir] == FREE) ){
                            temp = capsq + dir;
                            // add to movelist
                            move = orgmove;
                            move.path[move.l - 1] = temp;
                            m = SHIFT_WK|temp;
                            move.m[1] = m;
                            m = (b[capsq]<<6)|capsq;
                            move.m[move.l] = m;
                            move.l++;
                            found_cap++;
                            movelist[*n] = move;
                            b[capsq]--;
                            // further 1  jump
                            next_dir = 1 << ( 3 - i );
                            white_king_capture(b, n, movelist, temp,next_dir);
                            b[capsq]++;
                               } // if
                            else{
                              temp = capsq + dir;
                              do{
                              // add to movelist
                              move = orgmove;
                              move.path[move.l - 1] = temp;
                              m = SHIFT_WK|temp;
                              move.m[1] = m;
                              m = (b[capsq]<<6)|capsq;
                              move.m[move.l] = m;
                              move.l++;
                              found_cap++;
                              movelist[*n] = move;
                              (*n)++;
                              temp = temp + dir;
                                  } while ( b[temp] == FREE );
                                  }
                              }
                  }
        } // for

     if ( found_cap == 0 ) (*n)++;
 }


static void white_man_capture( int *b,int *n, struct move2 *movelist ,int j,int in_dir){
   //
   int dir,found_cap=0,sq1,sq2;
   unsigned i,m;
   struct move2 move,orgmove;
                    for ( i = 0; i < 4; i++ ){ // scan all 4 directions
                             dir = directions[i]; // -5,-4,5,4
                             if ( dir == in_dir ) continue;
                             sq1 = j + dir;
                             if (  ( b[sq1] & BLACK ) != 0 ){
                             sq2 = j + (dir<<1);
                             if ( b[sq2] == FREE ){
                                    // add to movelist
                                    if ( found_cap == 0 ) orgmove = movelist[*n];
                                    move = orgmove;
                                    move.path[move.l - 1] = sq2;
                                    m = (b[sq1]<<6)|sq1;
                                    move.m[move.l] = m;
                                    move.l++;
                             if ( sq2 < 10 ){  // promotion
                              	    m = SHIFT_WK|sq2;
                                   move.m[1] = m;
                                   found_cap++;
                                   movelist[*n] = move;
                                   
                                   if ( (sq2 == 5) || (sq2 == 8) ){
                                   (*n)++;
                                   continue;
                                                                                   }
                                   else{
                                   dir = ( i == 1 ) ? 4:8;
                                   b[sq1]--;
                                   //
                                   white_king_capture(b, n, movelist,sq2,dir);
                                   b[sq1]++;
                                         }
                                                 } // promotion
                                  else{  // non-promotion
                                  	m = SHIFT_WM|sq2;
                                    move.m[1] = m;
                                    found_cap++;
                                    movelist[*n] = move;
                                    b[sq1]--;
                                    //
                                    white_man_capture(b, n, movelist,sq2,-dir);
                                    b[sq1]++;
                                         }
                                   }
                               }
                          } // for
       if ( found_cap == 0 ) (*n)++;
   }


static int Test_From_pb( int *b, int temp, int dir){
// test if there is capture in perpendicular direction to dir from square
// for black color
int d,square;

       if ((dir&1) == 0)
         d = 5;
      else
         d = 4;

      square = temp;

          do{
            square = square + d;
              }while ( b[square] == FREE );
           if ( ( b[square] & WHITE ) != 0 )
           if ( b[square + d] == FREE )
           return (1);

     square = temp;
     d = -d; // another perp. direction
          do{
            square = square + d;
              }while ( b[square] == FREE );
           if ( ( b[square] & WHITE ) != 0 )
           if ( b[square + d] == FREE )
           return (1);
    return (0);
 }


static int Test_From_cb( int *b,int temp,int dir){
// test if there is capture in current direction dir
// for black color
          do{
            temp = temp + dir;
              }while ( b[temp] == FREE );
           if ( ( b[temp] & WHITE ) != 0 )
           if ( b[temp + dir] == FREE )
           return (1);
           return (0);
 }


static int Test_From_pw( int *b,int temp,int dir){
// test if there is capture in perpendicular direction to dir from square
// for white color
int d,square;

       if ((dir&1) == 0)
         d = 5;
       else
         d = 4;

       square = temp;

          do{
            square = square + d;
              }while ( b[square] == FREE );
           if ( ( b[square] & BLACK ) != 0 )
           if ( b[square + d] == FREE )
           return (1);

       square = temp;
       d = -d; // another perp. direction

          do{
            square = square + d;
              }while ( b[square] == FREE );
           if ( ( b[square] & BLACK ) != 0 )
           if ( b[square + d] == FREE )
           return (1);
    return (0);
 }


static int Test_From_cw( int *b,int temp,int dir){
// test if there is capture in current direction dir from square
// for white color
          do{
            temp = temp + dir;
              }while ( b[temp] == FREE );
           if ( ( b[temp] & BLACK ) != 0 )
           if ( b[temp + dir] == FREE )
           return (1);
           return (0);
 }


static unsigned int  Test_Capture( int *b, int color){
     //
        unsigned int square;
           if ( color == WHITE ){
           for( unsigned register i = 1;i <= num_wpieces;i++ ){
           if (( square = p_list[WHITE][i] ) == 0 ) continue;
           if ( (b[square] & MAN) != 0 ){
                       if ( square > 13 ){
                       if( (b[square-4] & BLACK) !=0)
                       if( b[square-8] == FREE )
                           return(i);
                       if( (b[square-5] & BLACK) !=0)
                       if( b[square-10] == FREE )
                           return(i);
                                                 }
                       if ( square < 32 ){
                       if( (b[square+4] & BLACK) !=0)
                       if( b[square+8] == FREE )
                           return(i);
                       if( (b[square+5] & BLACK) !=0)
                       if( b[square+10] == FREE )
                           return(i);
                                                   }
                                                            }
             else{  // KING
                if ( square < 32 ){
                if ( Test_From_cw(b,square,4) ) return (i);
                if ( Test_From_cw(b,square,5) ) return (i);
                                           }
                if ( square > 13 ){
                if ( Test_From_cw(b,square,-4) ) return (i);
                if ( Test_From_cw(b,square,-5) ) return (i);
                                            }
                   }
                                                          } // for
                return (0);
                         } // if ( color == WHITE )
        else{
            for( unsigned register i = 1;i <= num_bpieces;i++){
            if (( square = p_list[BLACK][i] )  == 0 ) continue;
                       if ( (b[square] & MAN) != 0 ){
                       if ( square < 32 ){
                       if( (b[square+4] & WHITE) !=0)
                       if( b[square+8] == FREE )
                           return(i);
                       if( (b[square+5] & WHITE) !=0)
                       if( b[square+10] == FREE )
                           return(i);
                                                  }
                       if ( square > 13 ){
                       if( (b[square-4] & WHITE) !=0)
                       if( b[square-8] == FREE )
                           return(i);
                       if( (b[square-5] & WHITE) !=0)
                       if( b[square-10] == FREE )
                           return(i);
                                                     }
                                                          }
            else{ // KING
                  if ( square < 32 ){
                  if ( Test_From_cb(b,square,4) ) return (i);
                  if ( Test_From_cb(b,square,5) ) return (i);
                                              }
                  if ( square > 13 ){
                  if ( Test_From_cb(b,square,-4) ) return (i);
                  if ( Test_From_cb(b,square,-5) ) return (i);
                                              }
                  }
                           } // for
                return (0);	
          } // if ( color == BLACK )
     }
     
     
static unsigned int Gen_Captures( int *b,struct move2 *movelist, int color,unsigned start){
int dir,next_dir,sq1,sq2,n=0,temp;
unsigned capsq,found_pd,i;
unsigned int m;
unsigned int j;
      if ( color == WHITE ){
        for ( unsigned register square=start; square <= num_wpieces;square++ ){
          if (( j = p_list[WHITE][square] ) == 0 ) continue;
          if ( ( b[j] & MAN ) != 0 ){
               b[j] = FREE;
               for ( i = 0; i < 4; i++ ){          // scan all 4 directions
                     dir = directions[i];             // dir = -5,-4,5,4
                     sq1 = j + dir;
                     if ( ( b[sq1] & BLACK ) != 0 ){
                     sq2 = j + (dir<<1);
                     if ( b[sq2] == FREE ){
                         // add to movelist
                         movelist[n].l = 3;
                         movelist[n].path[1] = sq2;
                         m = SHIFT_WM|j;
                         movelist[n].m[0] = m;
                         m = (b[sq1]<<6)|(sq1);
                         movelist[n].m[2] = m;
                         if ( sq2 < 10 ){ // promotion
                         m = SHIFT_WK|sq2;
                         movelist[n].m[1] = m;
                         if ( (sq2 == 5) || (sq2 == 8) ){
                         n++;
                         continue;
                                                                        }
                         else{
                         next_dir = (i == 1) ? 4:8; // in binary form 0100:1000
                         b[sq1]--;
                         // assert dir != 4 && dir != 5
                         white_king_capture(b,&n,movelist,sq2,next_dir);
                         b[sq1]++;
                                }
                                              }
                         else{ // non-promotion
                         m = SHIFT_WM|sq2;
                         movelist[n].m[1] = m;
                         b[sq1]--;
                         //next_dir = -dir;
                         white_man_capture(b, &n, movelist,sq2,-dir);
                         b[sq1]++;        	
                            }
                           } // if
                            } // if
                        } // for
                        b[j] =WHT_MAN;
               }    // if MAN

              else{ // b[j] is a KING
                         b[j] = FREE;
                         for ( i = 0; i < 4; i++ ){ // scan all 4 directions
                               dir = directions[i];
                               temp = j; // from square
                               do  temp += dir;while ( b[temp] == FREE );
                               if ( ( b[temp] & BLACK ) != 0 ){   
                               temp = temp + dir;
                               if ( b[temp] ) continue;
                               capsq = temp - dir;
                               found_pd = 0;
                               do{
                                if ( Test_From_pw(b,temp,dir) ){
                                   found_pd++;
                                   // add to movelist
                                   movelist[n].l = 3;
                                   movelist[n].path[1] = temp;
                                   m = SHIFT_WK|temp;
                                   movelist[n].m[1] = m;
                                   m = SHIFT_WK|j;
                                   movelist[n].m[0] = m;
                                   m = ( b[capsq]<<6 )|capsq;
                                   movelist[n].m[2] = m;
                                   b[capsq]--;
                                   // further jumps
                                   switch (i){
           case 0:next_dir = ( found_pd == 1 ) ? 13:5;break; // in binary form 1101:0101
           case 1:next_dir = ( found_pd == 1 ) ? 14:10;break; // in binary form 1110:1010
           case 2:next_dir = ( found_pd == 1 ) ? 7:5;break; // in binary form 0111:0101
           case 3:next_dir = ( found_pd == 1 ) ? 11:10;break; // in binary form 1011:1010
                                              }
                                   white_king_capture(b, &n, movelist,temp,next_dir);
                                   b[capsq]++;
                                                   } // if
                                   temp = temp + dir;
                                   } while ( b[temp] == FREE );

                                   if ( found_pd == 0 ){
                                   if ( (b[temp] & BLACK) != 0 && (b[temp + dir] == FREE) ){
                                      temp = capsq + dir;
                                      // add to movelist
                                      movelist[n].l = 3;
                                      movelist[n].path[1] = temp;
                                      m = SHIFT_WK|temp;
                                      movelist[n].m[1] = m;
                                      m = SHIFT_WK|j;
                                      movelist[n].m[0] = m;
                                      m = (b[capsq]<<6)|capsq;
                                      movelist[n].m[2] = m;
                                      b[capsq]--;
                                      // further 1 jump
                                      next_dir = 1 << ( 3 - i );
                                      white_king_capture(b,&n, movelist,temp,next_dir);
                                      b[capsq]++;
                                                                                                                  } // if
                                 else{
                                 temp = capsq + dir;
                                 do{
                                 // add to movelist
                                 movelist[n].l = 3;
                                 movelist[n].path[1] = temp;
                                 m = SHIFT_WK|temp;
                                 movelist[n].m[1] = m;
                                 m = SHIFT_WK|j;
                                 movelist[n].m[0] = m;
                                 m = (b[capsq]<<6)|capsq;
                                 movelist[n].m[2] = m;
                                 n++;
                                 temp = temp + dir;
                                      }while ( b[temp] == FREE );
                                        }
                                              }
               }
           } // for
     b[j] = WHT_KNG;
           } // else
                      } // for
                             }
      else{
        for ( unsigned register square=start; square <= num_bpieces;square++ ){
                if (( j = p_list[BLACK][square] ) == 0 ) continue;
                if ( (b[j] & MAN ) != 0 ){ 
                b[j] = FREE;
                for ( i = 0; i < 4; i++ ){  // scan all 4 directions
                        dir = directions[i];
                        sq1 = j + dir;
                        if ( ( b[sq1] & WHITE ) != 0 ){ 
                        sq2 = j + (dir<<1);
                        if ( b[sq2] == FREE ){
                        // add to movelist
                        movelist[n].l = 3;
                        movelist[n].path[1] = sq2;
                        m = SHIFT_BM|j;
                        movelist[n].m[0] = m;
                        m = (b[sq1]<<6)|sq1;
                        movelist[n].m[2] = m;
                        if ( sq2 > 35 ){ // promotion
                        m = SHIFT_BK|sq2;
                        movelist[n].m[1] = m;
                        if ( (sq2 == 37) || (sq2 == 40) ){
                        n++;
                        continue;
                                                                       }
                        else{
                        //next_dir = (dir == 4)?1:2; // in binary form 0001:0010
                        next_dir = dir - 3;
                        b[sq1]++;
                        // assert dir != -4 dir != -5 because can't promote capturing backwards
                        black_king_capture(b, &n, movelist,sq2,next_dir);
                        b[sq1]--;
                              }
                                           }
                        else{ // non-promotion
                        m = SHIFT_BM|sq2;
                        movelist[n].m[1] = m;
                        b[sq1]++;
                        //next_dir = -dir;
                        black_man_capture(b, &n, movelist,sq2,-dir);
                        b[sq1]--;
                                }
                             } 
                          }
                      }
                  b[j] = BLK_MAN;
        }
             else{     // b[j] is a KING
                      b[j] = FREE;
                      for ( i = 0; i < 4; i++ ){          // scan all 4 directions
                           dir = directions[i];
                           temp = j; // from square
                           do temp += dir;while ( b[temp] == FREE );
                           if ( ( b[temp] & WHITE ) != 0 ){       
                           temp = temp + dir;
                           if ( b[temp] ) continue;
                           capsq = temp - dir;
                           found_pd = 0;
                           do{
                           if ( Test_From_pb(b,temp,dir) ){
                             found_pd++;
                             // add to movelist
                             movelist[n].l = 3;
                             movelist[n].path[1] = temp;
                             m = SHIFT_BK|temp;
                             movelist[n].m[1] = m;
                             m = SHIFT_BK|j;
                             movelist[n].m[0] = m;
                             m = (b[capsq]<<6)|capsq;
                             movelist[n].m[2] = m;
                             b[capsq]++;
                             // further jumps
                             switch (i){
                              case 0:next_dir = ( found_pd == 1 ) ? 13:5;break; // in binary form 1101:0101
                              case 1:next_dir = ( found_pd == 1 ) ? 14:10;break; // in binary form 1110:1010
                              case 2:next_dir = ( found_pd == 1 ) ? 7:5;break; // in binary form 0111:0101
                              case 3:next_dir = ( found_pd == 1 ) ? 11:10;break; // in binary form 1011:1010
                                           }
                             black_king_capture(b, &n, movelist,temp,next_dir);
                             b[capsq]--;
                                                                    } // if
                             temp = temp + dir;
                                } while ( b[temp] == FREE );

                         if ( found_pd == 0 ){
                         if ( (b[temp] & WHITE) != 0 && (b[temp + dir]==FREE) ){
                             temp = capsq + dir;
                             // add to movelist
                             movelist[n].l = 3;
                             movelist[n].path[1] = temp;
                             m = SHIFT_BK|temp;
                             movelist[n].m[1] = m;
                             m = SHIFT_BK|j;
                             movelist[n].m[0] = m;
                             m = (b[capsq]<<6)|capsq;
                             movelist[n].m[2] = m;
                             b[capsq]++;
                             // further 1 jump
                             next_dir = 1 << ( 3 - i );
                             black_king_capture(b, &n, movelist,temp,next_dir);
                             b[capsq]--;
                                                                                                      } // if
                             else{
                                 temp = capsq + dir;
                                 do{
                                 // add to movelist
                                 movelist[n].l = 3;
                                 movelist[n].path[1] = temp;
                                 m = SHIFT_BK|temp;
                                 movelist[n].m[1] = m;
                                 m = SHIFT_BK|j;
                                 movelist[n].m[0] = m;
                                 m = (b[capsq]<<6)|capsq;
                                 movelist[n].m[2] = m;
                                 n++;
                                 temp = temp + dir;
                                    }while ( b[temp]==FREE );
                                    }
                                        }
                      }    
               } // for
           b[j] = BLK_KNG;
                           } // else
                     }
            }
  return (n);  // returns number of captures n
}

static unsigned int Gen_Moves( int *b, struct move2 *movelist,int color ){
   //
   unsigned int m,n=0,square,temp;
         if ( color == WHITE ){
         for ( unsigned register i = 1;i <= num_wpieces;i++){
                 if (( square = p_list[WHITE][i] ) == 0) continue;
                        if ( ( b[square] & MAN ) != 0 ){
                                    temp = square - 5;
                                    if ( b[temp] == FREE ){
                                            movelist[n].l = 2;
                                            if ( square < 14 )
                                            m = SHIFT_WK|temp;
                                            else
                                            m = SHIFT_WM|temp;
                                            movelist[n].m[1] = m;
                                            m = SHIFT_WM|square;
                                            movelist[n].m[0] = m;
                                            n++;
                                                                          }
                                   temp++;
                                   if ( b[temp] == FREE ){
                                           movelist[n].l = 2;
                                           if ( square < 14 )
                             	            m = SHIFT_WK|temp;
                             	            else
                                            m = SHIFT_WM|temp;
                                            movelist[n].m[1] = m;
                                            m = SHIFT_WM|square;
                                            movelist[n].m[0] = m;
                                            n++;
                                                                         }                                                                
                                                 } // MAN
                           else{ // KING
                             temp = square + 4;
                             while  ( b[temp] == FREE ){
                             movelist[n].l = 2;
                             m = SHIFT_WK|temp;
                             movelist[n].m[1] = m;
                             m = SHIFT_WK|square;
                             movelist[n++].m[0] = m;
                             temp += 4;
                                                        } // while
                             temp = square + 5;
                             while  ( b[temp] == FREE ){
                             movelist[n].l = 2;
                             m = SHIFT_WK|temp;
                             movelist[n].m[1] = m;
                             m = SHIFT_WK|square;
                             movelist[n++].m[0] = m;
                             temp += 5;
                                                        } // while
                             temp = square - 4;
                             while  ( b[temp] == FREE ){
                             movelist[n].l = 2;
                             m = SHIFT_WK|temp;
                             movelist[n].m[1] = m;
                             m = SHIFT_WK|square;
                             movelist[n++].m[0] = m;
                             temp -= 4;
                                                        } // while
                             temp = square - 5;
                             while  ( b[temp] == FREE ){
                             movelist[n].l = 2;
                             m = SHIFT_WK|temp;
                             movelist[n].m[1] = m;
                             m = SHIFT_WK|square;
                             movelist[n++].m[0] = m;
                             temp -= 5;
                                                        } // while
                               } // else
                  } // for
            } // color == WHITE
         else{
          for ( unsigned register i = 1;i <= num_bpieces;i++){
          if (( square = p_list[BLACK][i] ) == 0) continue;
                 if ( ( b[square] & MAN ) != 0 ){
                  	      temp = square + 4;
                            if ( b[temp] == FREE ){
                               movelist[n].l = 2;
                               if ( square > 31 )
                           	m =  SHIFT_BK|temp;
                           	else
                               	m = SHIFT_BM|temp;
                               movelist[n].m[1] = m;
                               m = SHIFT_BM|square;
                               movelist[n].m[0] = m;
                               n++;
                                              }
                            temp++;          
                            if ( b[temp] == FREE ){
                               movelist[n].l = 2;
                               if ( square > 31 )
                  	         m = SHIFT_BK|temp;
                  	         else
                               	m = SHIFT_BM|temp;
                               movelist[n].m[1] = m;
                               m = SHIFT_BM|square;
                               movelist[n].m[0] = m;
                               n++;
                                 }
                                             } // MAN
                 else{   // KING
                        temp = square + 4;
                        while  ( b[temp] == FREE ){
                        movelist[n].l = 2;
                        m = SHIFT_BK|temp;
                        movelist[n].m[1] = m;
                        m = SHIFT_BK|square;
                        movelist[n++].m[0] = m;
                        temp += 4;
                                           } // while
                        temp = square + 5;
                        while  ( b[temp] == FREE ){
                        movelist[n].l = 2;
                        m = SHIFT_BK|temp;
                        movelist[n].m[1] = m;
                        m = SHIFT_BK|square;
                        movelist[n++].m[0] = m;
                        temp += 5;
                                            } // while
                        temp = square - 4;
                        while  ( b[temp] == FREE ){
                        movelist[n].l = 2;
                        m = SHIFT_BK|temp;
                        movelist[n].m[1] = m;
                        m = SHIFT_BK|square;
                        movelist[n++].m[0] = m;
                        temp -= 4;
                                           } // while
                        temp = square - 5;
                        while  ( b[temp] == FREE ){
                        movelist[n].l = 2;
                        m = SHIFT_BK|temp;
                        movelist[n].m[1] = m;
                        m = SHIFT_BK|square;
                        movelist[n++].m[0] = m;
                        temp -= 5;
                                            } // while
                        }  // else
                               } // for
          }
    return (n); // returns number of moves n
}

static unsigned int Gen_Proms( int *b,struct move *movelist,int color){
   // generates only promotions
   // used in quiescent search
   unsigned int n = 0;
   
         if ( color == WHITE ){
          if ( (b[13] & WHITE) != 0 ){
          if ( (b[13] & MAN) != 0 ){
          if ( b[8] == FREE ){
          movelist[n].m[1] = 0x248;
          movelist[n++].m[0] = 0x14D;
                                        }
                                                   }
                                                       }
                        
          if ( (b[10] & WHITE) != 0 ){
          if ( (b[10] & MAN) != 0 ){
          if ( b[5] == FREE ){
          movelist[n].m[1] = 0x245;
          movelist[n++].m[0] = 0x14A;
                                         }
          if ( b[6] == FREE ){
          movelist[n].m[1] = 0x246;
          movelist[n++].m[0] = 0x14A;
                                         }
               }
                        }
                        
          if ( (b[11] & WHITE) != 0 ){
          if ( (b[11] & MAN) != 0 ){
          if ( b[6] == FREE ){
          movelist[n].m[1] = 0x246;
          movelist[n++].m[0] = 0x14B;
                                         }
          if ( b[7] == FREE ){
          movelist[n].m[1] = 0x247;
          movelist[n++].m[0] = 0x14B;
                           }              
               }
                        }
                        
          if ( (b[12] & WHITE) != 0 ){
          if ( (b[12] & MAN) != 0 ){
          if ( b[7] == FREE ){
          movelist[n].m[1] = 0x247;
          movelist[n++].m[0] = 0x14C;
                           }
          if ( b[8] == FREE ){
          movelist[n].m[1] = 0x248;
          movelist[n++].m[0] = 0x14C;
                           }              
               }
                   }                     
               }
         else{
         if ( (b[32] & BLACK) != 0){
         if ( (b[32] & MAN) != 0){
         if ( b[37] == FREE ){
         movelist[n].m[1] = 0x2A5;
         movelist[n++].m[0] = 0x1A0;
                          }
                        }
                      }
                      
         if ( (b[33] & BLACK) != 0){
         if ( (b[33] & MAN) != 0){
         if ( b[37] == FREE ){
         movelist[n].m[1] = 0x2A5;
         movelist[n++].m[0] = 0x1A1;
                                          }
         if ( b[38] == FREE ){
         movelist[n].m[1] = 0x2A6;
         movelist[n++].m[0] = 0x1A1;
                                           }
                            }
                      }
                      
         if ( (b[34] & BLACK) != 0){
         if ( (b[34] & MAN) != 0){
         if ( b[38] == FREE ){
         movelist[n].m[1] = 0x2A6;
         movelist[n++].m[0] = 0x1A2;
                          }
         if ( b[39] == FREE ){
         movelist[n].m[1] = 0x2A7;
         movelist[n++].m[0] = 0x1A2;
                          }          
                            }
                      }

         if ( (b[35] & BLACK) != 0){
         if ( (b[35] & MAN) != 0){
         if ( b[39] == FREE ){
         movelist[n].m[1] = 0x2A7;
         movelist[n++].m[0] = 0x1A3;
                          }
         if ( b[40] == FREE ){
         movelist[n].m[1] = 0x2A8;
         movelist[n++].m[0] = 0x1A3;
                          }          
                            }
                      }
         }
  return (n); // returns number of promotions n
}


static void domove(int *b,struct move2 *move,int stm)
/*----> purpose: execute move on board and update HASH_KEY */
{
   unsigned int contents,from,target;
 
   HASH_KEY ^= HashSTM;

   from = ((move->m[0]) & 0x3f);
   b[from] = FREE;
   contents = ((move->m[0]) >> 6);
   Reversible[realdepth] = ((contents & KING) && (move->l == 2));

   HASH_KEY ^= ZobristNumbers[from][contents];
   g_pieces[contents]--;

   target = (( move->m[1]) & 0x3f);
   contents = ((move->m[1]) >> 6);
   HASH_KEY ^= ZobristNumbers[target][contents];
   b[target] = contents;
   g_pieces[contents]++;

   indices[target] = indices[from];
   p_list[stm][indices[target]] = target;

      for(unsigned register i=(move->l)-1;i>1;i--){
      target = ((move->m[i]) & 0x3f);
      b[target] = FREE;
      contents = ((move->m[i]) >> 6);
      HASH_KEY ^= ZobristNumbers[target][contents];
      g_pieces[contents]--;
      c_num[realdepth][i] = indices[target];
      p_list[(stm^CC)][indices[target]] = 0;
                         }
      realdepth++;
}

static void domove2(int *b,struct move2 *move,int stm )
/*----> purpose: execute move on board without HASH_KEY updating */
   {
   unsigned int contents,from,target;
   from = ((move->m[0]) & 0x3f);
   b[from] = FREE;
   contents = ((move->m[0]) >> 6);
   // Reversible[realdepth] = ((contents & KING) && (move->l == 2));
   g_pieces[contents]--;
   target = ((move->m[1]) & 0x3f);
   contents = ((move->m[1]) >> 6);
   b[target] = contents;
   g_pieces[contents]++;
   indices[target] = indices[from];
   p_list[stm][indices[target]] = target;
   for(unsigned register i=(move->l)-1;i>1;i--){
      target = ((move->m[i]) & 0x3f);
      c_num[realdepth][i] = indices[target];
      p_list[(stm^CC)][indices[target]] = 0;
      contents = ((move->m[i]) >> 6);
      g_pieces[contents]--;
      b[target] = FREE;
                                      }
      realdepth++;
}

static void __inline doprom(int *b,struct move *move,int stm )
/*----> purpose: execute promotion on board and update HASH_KEY */
{
   unsigned int contents,from,target;
   HASH_KEY ^= HashSTM;
   from = ((move->m[0]) & 0x3f);
   b[from] = FREE;
   contents = ((move->m[0]) >> 6);
   HASH_KEY ^= ZobristNumbers[from][contents];
   // Reversible[realdepth] = ((contents & KING) && (move->l == 2));
   g_pieces[contents]--;

   target = ((move->m[1]) & 0x3f);
   contents = ((move->m[1]) >> 6);
   HASH_KEY ^= ZobristNumbers[target][contents];
   b[target] = contents;
   
   g_pieces[contents]++;

   indices[target] = indices[from];
   p_list[stm][indices[target]] = target;
   realdepth++;
}


static void __inline undoprom(int *b,struct move *move,int stm )
/*----> purpose: undo what doprom did */
{
   unsigned int contents,from,to;
 
   realdepth--;
   to = ((move->m[1]) & 0x3f); // to
   b[to] = FREE;
   contents = ((move->m[1]) >> 6); // contents
   g_pieces[contents]--;

   from = ((move->m[0]) & 0x3f); // from
   contents = ((move->m[0]) >> 6); // contents
   b[from] = contents;
   g_pieces[contents]++;

   indices[from] = indices[to];
   p_list[stm][indices[from]] = from;
}


static void undomove(int *b,struct move2 *move,int stm )
 /*----> purpose: undo what domove did */
   {
   unsigned int contents,from,to;
   realdepth--;
   to = ((move->m[1]) & 0x3f); // to
   b[to] = FREE;
   contents = ((move->m[1]) >> 6); // contents
   g_pieces[contents]--;
   from = ((move->m[0]) & 0x3f); // from
   contents = ((move->m[0]) >> 6); // contents
   b[from] = contents;
   g_pieces[contents]++;
   indices[from] = indices[to];
   p_list[stm][indices[from]] = from;
   for(unsigned register i=(move->l)-1;i>1;i--){
      contents = ((move->m[i]) >> 6);
      g_pieces[contents]++;
      to = ((move->m[i]) & 0x3f);
      b[to] = contents;
      indices[to] = c_num[realdepth][i];
      p_list[(stm^CC)][indices[to]] = to;
         }
}

static int evaluation(int b[46], int color, int alpha, int beta){
   /*----> purpose: static evaluation of the board */
   //U64 TESTHASH = Position_to_Hashnumber(b,color);
   //assert( TESTHASH == HASH_KEY );
   int eval;
   int GLAV = 0;
   if (( ( HASH_KEY ^ EvalHash[ ( U32) (HASH_KEY & EC_MASK )  ] ) & 0xffffffffffff0000 ) == 0 )
       {
   eval  = (int) ((S16) ( EvalHash[ (U32) (HASH_KEY & EC_MASK ) ] & 0xffff ) );
   return  eval;
       }
   int nbm = g_pieces[BLK_MAN]; // number of black men
   int nwm = g_pieces[WHT_MAN]; // number of white men
   int nbk = g_pieces[BLK_KNG]; // number of black kings
   int nwk = g_pieces[WHT_KNG]; // number of white kings

   if ( (nbm == 0) && (nbk == 0) ){
   eval = realdepth - MATE;
   //	assert( color == BLACK );
   //EvalHash[ (U32) ( HASH_KEY & EC_MASK ) ] = (HASH_KEY & 0xffffffffffff0000) | ( eval & 0xffff);
   return eval;
                                                       }
   if ( (nwm == 0) && (nwk == 0) ){
   eval = realdepth - MATE;
   //	assert( color == WHITE );
   //EvalHash[ (U32) ( HASH_KEY & EC_MASK ) ] = (HASH_KEY & 0xffffffffffff0000) | ( eval & 0xffff);
   return eval;
                                                          }

           int White = nwm + nwk; // total number of white pieces
           int Black = nbm + nbk;     // total number of black pieces
           int v1 = 100 * nbm + 300 * nbk;
           int v2 = 100 * nwm + 300 * nwk;
           eval = v1 - v2;         // material values
     
           // draw situations
           if ( nbm == 0 && nwm == 0 && abs( nbk - nwk) <= 1 ){ 
           EvalHash[ (U32) ( HASH_KEY & EC_MASK ) ] = (HASH_KEY & 0xffffffffffff0000) | ( 0 & 0xffff);
           return (0); // only kings left
                      }
           if ( ( eval > 0 ) && ( nwk > 0 ) && (Black < (nwk+2)) ){
           EvalHash[ (U32) ( HASH_KEY & EC_MASK ) ] = (HASH_KEY & 0xffffffffffff0000) | ( 0 & 0xffff);
           return (0); // black cannot win
                                          }

           if ( ( eval < 0 ) && (nbk > 0) && (White < (nbk+2)) ){
           EvalHash[ (U32) ( HASH_KEY & EC_MASK ) ] = (HASH_KEY & 0xffffffffffff0000) | ( 0 & 0xffff);
           return (0); //  white cannot win
                                           }

  static U8 PST_man_op[41] = {0,0,0,0,0,   // 0 .. 4
                             15,40,42,45,0,              // 5 .. 8 (9)
                             12,38,36,15,                     // 10 .. 13
                             28,26,30,20,0,               // 14 .. 17 (18)
                             18,26,36,28,                    // 19 .. 22
                             32,38,10,18,0,                // 23 .. 26 (27)
                             18,22,24,20,                 //  28 .. 31
                             26,0,0,0,0,                      // 32 .. 35 (36)
                             0,0,0,0                          // 37 .. 40
                                       };
                                       
 static U8 PST_man_en[41] = {0,0,0,0,0,  // 0 .. 4
                             0,2,2,2,    0,                  // 5 .. 8 (9)
                             4,4,4,4,                     // 10 .. 13
                             6,6,6,6,    0,               // 14 .. 17 (18)
                             10,10,10,10,                  // 19 .. 22
                             16,16,16,16,   0,              // 23 .. 26 (27)
                             22,22,22,22,                //  28 .. 31
                             30,0,0,0,         0,            // 32 .. 35 (36)
                             0,0,0,0                        // 37 .. 40
                                       };     
 
  static U8 PST_king[41] = {0,0,0,0,0,  // 0..4
                                               20,5,0,10,0, // 5..8 (9)
                                               20,5,10,10, // 10..13
                                               5,20,12,10,0, // 14..18
                                               5,20,12,0, // 19..22
                                               0,12,20,5,0, // 23..27
                                               10,12,20,5, // 28..31
                                               10,10,5,20,0, // 32..36
                                               10,0,5,20 // 37..40
                                              };
           unsigned int i;       
           int square;  
           int opening = 0;
           int endgame = 0;
           //a piece of code to encourage exchanges
		  //in case of material advantage:
		      // king's balance
            if ( nbk != nwk){
            if ( nwk == 0 ){
            	    if ( nwm <= 4 ){
                 endgame += 50;
                 if ( nwm <= 3 ){
                 endgame += 100;
                 if ( nwm <= 2 ){
                 endgame += 100;
                 if ( nwm <= 1 )
                 endgame += 100;	
                                          }
                                        }
                                       }
                                      }
             if ( nbk == 0 ){
            	 if ( nbm <= 4 ){
                 endgame -= 50;
                 if ( nbm <= 3 ){
                 endgame -= 100;
                 if ( nbm <= 2 ){
                 endgame -= 100;
                 if ( nbm <= 1 )
                 endgame -= 100;	
                                          }
                                        }
                                      }
                                  }  
                         } 
           else{           
           if ( (nbk == 0) && (nwk == 0) )
           eval += 250*( v1 - v2 ) / ( v1 + v2 ); 
           if ( nbk + nwk != 0 )
           eval += 100*( v1 - v2 ) / ( v1 + v2 );
                 }
         
           // special case : very very late endgame
           if ( (White < 4) && (Black < 4) ){
           GLAV = 0; // main diagonal a1-h8 control
           for (  i = 5; i < 41; i+= 5 )
           GLAV += b[i];
           if ( eval < 0 && nbk == 1 && GLAV == BLK_KNG ){
           if ( nbm == 0 || b[32] == BLK_MAN ){
           EvalHash[ (U32) ( HASH_KEY & EC_MASK ) ] = (HASH_KEY & 0xffffffffffff0000) | ( 0 & 0xffff);
           return (0);
                                            }
                                          }
          if ( eval > 0 && nwk == 1 && GLAV == WHT_KNG ){
          if ( nwm == 0 || b[13] == WHT_MAN ){
          EvalHash[ (U32) ( HASH_KEY & EC_MASK ) ] = (HASH_KEY & 0xffffffffffff0000) | ( 0 & 0xffff);
          return (0);
                                            }
                             }
          if ( (nwk != 0) && (nbk != 0) && ( nbm == 0 ) && ( nwm == 0 ) ){
          // only kings left
          if ( nbk == 1 && nwk == 3 ){
             int double_r1 = is_wht_kng_1(b); //
             int double_r2 = is_wht_kng_2(b); //
              if (( double_r1 < 3 ) && ( double_r2 < 3 )){
              if ( double_r1 + double_r2 == 3 )
              	eval -= 300;
              if ( double_r1 + double_r2 == 2 )
              	eval -= 300;
              if ( double_r1 + double_r2 == 1 )
              	eval -= 100;
                                                                                   }
           
              int triple_r1 = is_wht_kng_3(b); //
              int triple_r2 = is_wht_kng_4(b); //
              if ( ( triple_r1 < 3 ) && ( triple_r2 < 3 ) ){
              if ( triple_r1 + triple_r2 == 3 )
             	eval -= 200;
             	if ( triple_r1 + triple_r2 == 2 )
             	eval -= 200;
             	if ( triple_r1 + triple_r2 == 1 )
             	eval -= 100;
                                                                               }
               	if ( is_blk_kng_1(b) == 0 && is_blk_kng_2(b) == 0 ){
                  if ( is_blk_kng_3(b) == 0 && is_blk_kng_4(b) == 0 ){
                  if ( color == BLACK ){
                 	if (( b[15] == WHT_KNG ) && ( b[29] == WHT_KNG ) && ( b[16] == WHT_KNG ))
                 	eval -= 1000;
                 	if (( b[29] == WHT_KNG ) && ( b[16] == WHT_KNG ) && ( b[30] == WHT_KNG ))
                 	eval -= 1000;
                	if (( b[15] == WHT_KNG ) && ( b[24] == WHT_KNG ) && ( b[21] == WHT_KNG ))
                 	eval -= 1000;
                  if (( b[24] == WHT_KNG ) && ( b[21] == WHT_KNG ) && ( b[30] == WHT_KNG ))
                 	eval -= 1000;
                 	          }
                 	        }
                 	      }
       	          }
              if ( nwk == 1 && nbk == 3 ){
              int double_r1 = is_blk_kng_1(b); //
              int double_r2 = is_blk_kng_2(b); //
              if ( ( double_r1 < 3 ) && ( double_r2 < 3 ) ){
              if ( double_r1 + double_r2 == 3 )
              	eval += 300;
              if ( double_r1 + double_r2 == 2 )
              	eval += 300;
               if ( double_r1 + double_r2 == 1 )
              	eval += 100;
                                                                                     }
              int triple_r1 = is_blk_kng_3(b); //
              int triple_r2 = is_blk_kng_4(b); //
              if ( ( triple_r1  < 3 ) && ( triple_r2 < 3 ) ){
              if ( triple_r1 + triple_r2 == 3 )
             	eval += 200;
             	if ( triple_r1 + triple_r2 == 2 )
             	eval += 200;
             if ( triple_r1 + triple_r2 == 1 )
             	eval += 100;
                                                                                 }
             	if ( is_wht_kng_1(b) == 0 && is_wht_kng_2(b) == 0 ){
              if ( is_wht_kng_3(b) == 0 && is_wht_kng_4(b) == 0 ){
              if ( color == WHITE ){
                 	if (( b[15] == BLK_KNG ) && ( b[29] == BLK_KNG ) && ( b[16] == BLK_KNG ))
                 	eval += 1000;
                 	if (( b[29] == BLK_KNG ) && ( b[16] == BLK_KNG ) && ( b[30] == BLK_KNG ))
                 	eval += 1000;
                	if (( b[15] == BLK_KNG ) && ( b[24] == BLK_KNG ) && ( b[21] == BLK_KNG ))
                 	eval += 1000;
                  if (( b[24] == BLK_KNG ) && ( b[21] == BLK_KNG ) && ( b[30] == BLK_KNG ))
                 	eval += 1000;
                 	          }
                 	        }
                 	      }
       	          }
          for ( i = 1;i <= num_bpieces;i++){
          if ( (square = p_list[BLACK][i] )  == 0 ) continue;
          if ( ( b[square] & KING ) != 0 ) // black king
          eval += PST_king[square];
                                                               } // for
                       
    for ( i = 1;i <= num_wpieces;i++){
           if ( (square = p_list[WHITE][i]) == 0 ) continue;
           if ( ( b[square] & KING ) != 0 ) // white king
           eval -= PST_king[square];
                                                           } // for
          // negamax formulation requires this:
          eval = ( color == BLACK ) ? eval : -eval;
          EvalHash[ (U32) ( HASH_KEY & EC_MASK ) ] = (HASH_KEY & 0xffffffffffff0000) | ( eval & 0xffff);
          return (eval); 
                              } // only kings left
    if ( nbk == 0 && nwk == 0 ){ // only men left
                // strong opposition
  	if ( b[19] == BLK_MAN )
    if ( (b[32] == WHT_MAN) && (b[28] == WHT_MAN ))
    if (( b[23] == FREE ) && ( b[24] == FREE ))
    	eval += 24;
     if ( b[26] == BLK_MAN )
     if ( (b[40] == WHT_MAN) && (b[35] == WHT_MAN ))
     if (( b[30] == FREE ) && ( b[31] == FREE ))
    eval += 24;
     	
     if ( b[26] == WHT_MAN )
     if ( (b[13] == BLK_MAN) && (b[17] == BLK_MAN ))
     if (( b[21] == FREE ) && ( b[22] == FREE ))
  	 eval -= 24;
   	if ( b[19] == WHT_MAN )
    if ( (b[5] == BLK_MAN) && (b[10] == BLK_MAN ))
    if (( b[14] == FREE ) && ( b[15] == FREE ))
	eval -= 24;
  
    // most favo(u)rable opposition
  
    if (( b[28] == BLK_MAN ) && ( b[37] == WHT_MAN ) && ( b[38] == FREE ))
    	if (( b[32] == FREE ) && ( b[33] == FREE ))
    eval += 28;
    if (( b[17] == WHT_MAN ) && ( b[8] == BLK_MAN ) && ( b[7] == FREE ))
    if (( b[12] == FREE ) && ( b[13] == FREE ))
    eval -= 28;
                                        } // only men left
                              } // special case : very very late endgame

 
     // piece-square-tables    
     
     for ( i = 1;i <= num_bpieces;i++){
          if ( (square = p_list[BLACK][i] )  == 0 ) continue;
          if ( ( b[square] & MAN ) != 0 ){ // black man
          opening += PST_man_op[square];
          endgame += PST_man_en[square];
                                                            }
                                                      } // for
                       
    for ( i = 1;i <= num_wpieces;i++){
           if ( (square = p_list[WHITE][i]) == 0 ) continue;
           if ( ( b[square] & MAN ) != 0 ){  // white man
           opening -= PST_man_op[ 45 - square];
           endgame -= PST_man_en[ 45 - square];
                                                              }
                                                  } // for
                                                          
   int phase = nbm + nwm - nbk - nwk;
   if ( phase < 0 ) phase = 0;
   int antiphase = 24 - phase;
      
   eval += (( opening * phase + endgame * (antiphase) )/24);
    if ( ( White + Black < 8 ) &&  nbk != 0 && nwk != 0 && nbm != 0 && nwm != 0 ){
    	if ( abs(nbm - nwm ) <= 1  && abs(nbk - nwk ) <= 1 && abs( White - Black ) <= 1 ){
           eval /= 2;
                          }
                      	}
    //Lazy evaluation
   // Early exit from evaluation if eval already is extremely low or extremely high
   if ( beta - alpha == 1 ){
   int teval = ( color == WHITE ) ? -eval : eval;
   if ( ( teval - 130 ) > beta )
   return teval;
   if ( ( teval + 130 ) < alpha )
   return teval;
                                       }

   static U8 edge1[4] = { 5, 14, 23, 32};
   static U8 edge2[3] = { 13, 22, 31};
   //                                          0   1    2    3    4     5    6     7     8   9  10  11 12
   static U8 edge_malus[13] = {0 ,60 , 40 ,30, 20 , 5 ,  5 ,   5 ,   0 , 0 , 0 ,   0 , 0 };
   int nbme = 0;
   int nwme = 0;
                /* men on edges */
   if ( nbm <= 4 && nwm <= 4 ){
   for( i = 0; i < 4; i++){
         if( b[edge1[i]] == BLK_MAN ){
         if(( b[edge1[i] + 5 ] == FREE ) && ( b[edge1[i] + 10 ] == WHT_MAN ))
         nbme++;
         else
         if( b[edge1[i] + 1 ] == WHT_MAN )
         nbme++;
                                                           }
         if( b[edge1[i]] == WHT_MAN ){
         if(( b[edge1[i] - 4 ] == FREE ) && ( b[edge1[i] - 8 ] == BLK_MAN ))
         nwme++;
         else
         if( b[edge1[i] + 1 ] == BLK_MAN )
         nwme++;
                                                              } 
                                 };
       for( i = 0; i < 3; i++){
         if( b[edge2[i]] == BLK_MAN ){
         if(( b[edge2[i] + 4 ] == FREE ) && ( b[edge2[i] + 8 ] == WHT_MAN ))
         nbme++;
         else
         if( b[edge2[i] - 1 ] == WHT_MAN )
         nbme++;
                                                           }
         if( b[edge2[i]] == WHT_MAN ){
         if(( b[edge2[i] - 5 ] == FREE ) && ( b[edge2[i] - 10 ] == BLK_MAN ))
         nwme++;
         else
         if( b[edge2[i] - 1 ] == BLK_MAN )
         nwme++;
                                                              } 
                                 };                            
                       }
   eval -= nbme*edge_malus[nbm];
   eval += nwme*edge_malus[nwm];
   // back rank ( a1,c1,e1,g1 ) guard
   // back rank values
   static S8 br[16] = {0,-1,1, 0,3,3,3,3,2,2,2,2,4,4,8,8  };

   int code;
   int backrank;
   code = 0;
   if(b[5] & MAN) code++;
   if(b[6] & MAN) code+=2;
   if(b[7] & MAN) code+=4; // Golden checker
   if(b[8] & MAN) code+=8;
   backrank = br[code];
   code = 0;
   if(b[37] & MAN) code+=8;
   if(b[38] & MAN) code+=4; // Golden checker
   if(b[39] & MAN) code+=2;
   if(b[40] & MAN) code++;
   backrank -= br[code];
   int brv = ( NOT_ENDGAME ? 2 : 1);  // multiplier for back rank -- back rank value
   eval += brv * backrank;

   opening = 0;
   endgame = 0;

   if ( (nbk == 0) && (nwk == 0) ){
    int j;	
   	               
    // the move : the move is an endgame term that defines whether one side
    // can force the other to retreat
    if ( nbm == nwm && nbm + nwm <= 12 ){
    int move;

    int stonesinsystem = 0;
    if ( color == BLACK && has_man_on_7th(b,BLACK) == 0 )
         {
    for ( i=5; i <= 8;i++)
             {
    for ( j=0; j < 4; j++)
                    {
           if ( b[i+9*j] != FREE ) stonesinsystem++;
                     }
              }
     if ( b[32] == BLK_MAN ) stonesinsystem++; // exception from the rule
     if ( stonesinsystem % 2 ) // the number of stones in blacks system is odd -> he has the move
     endgame += 4;
     else
     endgame -= 4;
        }
        
     if ( color == WHITE && has_man_on_7th(b , WHITE) == 0 )
     {
     for ( i=10; i <= 13;i++)
              {
     for ( j=0; j < 4; j++)
                  {
            if ( b[i+9*j] != FREE ) stonesinsystem++;
                   }
              }
     if ( b[13] == WHT_MAN ) stonesinsystem++;
     if ( stonesinsystem % 2 ) // the number of stones in whites system is odd -> he has the move
     endgame -= 4;
     else
     endgame += 4;
     }
                      }
             /* balance                */
            /* how equally the pieces are distributed on the left and right sides of the board */
    if ( nbm == nwm ){
    int balance = 0;
    int code;
    int index;
    static int value[7] = {0,0,0,0,0,1,256};
    int nbma,nbmb,nbmc,nbmd ; // number black men left a-b-c-d
    int nbme,nbmf,nbmg,nbmh ; // number black men right e-f-g-h
    int nwma,nwmb,nwmc,nwmd ; // number white men left a-b-c-d
    int nwme,nwmf,nwmg,nwmh;  // number white men right e-f-g-h
    // left flank
    code = 0;
    // count file-a men ( on 5,14,23,32 )
    code+=value[b[5]];
    code+=value[b[14]];
    code+=value[b[23]];
    code+=value[b[32]];
    nwma = code & 15;
    nbma = (code>>8) & 15;
    code = 0;
    // count file-b men ( on 10,19,28,37 )
    code+=value[b[10]];
    code+=value[b[19]];
    code+=value[b[28]];
    code+=value[b[37]];
    nwmb = code & 15;
    nbmb = (code>>8) & 15;
    
    code = 0;
    // count file-c men ( on 6,15,24,33 )
    code+=value[b[6]];
    code+=value[b[15]];
    code+=value[b[24]];
    code+=value[b[33]];
    nwmc = code & 15;
    nbmc = (code>>8) & 15;
    code = 0;
    // count file-d men ( on 11,20,29,38 )
    code+=value[b[11]];
    code+=value[b[20]];
    code+=value[b[29]];
    code+=value[b[38]];
    nwmd = code & 15;
    nbmd = (code>>8) & 15;
    
    // right flank
    code = 0;
    // count file-e men ( on 7,16,25,34 )
    code+=value[b[7]];
    code+=value[b[16]];
    code+=value[b[25]];
    code+=value[b[34]];
    nwme = code & 15;
    nbme = (code>>8) & 15;
    code = 0;
    // count file-f men ( on 12,21,30,39 )
    code+=value[b[12]];
    code+=value[b[21]];
    code+=value[b[30]];
    code+=value[b[39]];
    nwmf = code & 15;
    nbmf = (code>>8) & 15;
    code = 0;
    // count file-g men ( on 8,17,26,35 )
    code+=value[b[8]];
    code+=value[b[17]];
    code+=value[b[26]];
    code+=value[b[35]];
    nwmg = code & 15;
    nbmg = (code>>8) & 15;
    code = 0;
    // count file-h men ( on 13,22,31,40 )
    code+=value[b[13]];
    code+=value[b[22]];
    code+=value[b[31]];
    code+=value[b[40]];
    nwmh = code & 15;
    nbmh = (code>>8) & 15;
    
    // 2 blacks stops 3 whites in right flank
    if ( ( nwmf+nwmg+nwmh+nwme ) == 3 ){
    if ( ( nbmf+nbmg+nbmh+nbme ) == 2 ){
    if ( ( b[21] == BLK_MAN ) && ( b[22] == BLK_MAN ) ){
    if ( ( b[35] == WHT_MAN ) && ( b[30] == WHT_MAN ) && ( b[31] == WHT_MAN ) )
    endgame += 24;
                                                                                                 }
    if ( ( b[26] == WHT_MAN ) && ( b[30] == WHT_MAN ) && ( b[31] == WHT_MAN ) ){
    if ( ( b[22] == BLK_MAN ) && ( b[17] == BLK_MAN ) )
    endgame += 24;
                                                                                                               }
                                                        }
                                           } 

    // 2 blacks stops 3 whites in left flank
    int nbmabcd = nbma+nbmb+nbmc+nbmd;
    int nwmabcd = nwma+nwmb+nwmc+nwmd;
    if ( ( nbmabcd == 2 ) && ( nwmabcd == 3 ) ){
    if (( b[23] == BLK_MAN ) && ( b[20] == BLK_MAN ))
    if ( ( b[28] == WHT_MAN ) && ( b[32] == WHT_MAN ) && ( b[33] == WHT_MAN ) )
    endgame += 24;
    if (( b[14] == BLK_MAN ) && ( b[11] == BLK_MAN ))
    if ( ( b[19] == WHT_MAN ) && ( b[23] == WHT_MAN ) && ( b[24] == WHT_MAN ) )
    	endgame += 24;
                                                                               }
     // for white color
    // 2 whites stops 3 blacks
    if ( ( nwma+nwmb+nwmc+nwmd ) == 2 ){
    if ( ( nbma+nbmb+nbmc+nbmd ) == 3 ){
    if ( ( b[23] == WHT_MAN ) && ( b[24] == WHT_MAN ) ){
    if ( ( b[10] == BLK_MAN ) && ( b[14] == BLK_MAN ) && ( b[15] == BLK_MAN ) )
    endgame -= 24;
                                                                                                    }
    if ( ( b[23] == WHT_MAN ) && ( b[28] == WHT_MAN ) ){
    if ( ( b[14] == BLK_MAN ) && ( b[15] == BLK_MAN ) && ( b[19] == BLK_MAN ) )
    endgame -= 24;
                                                                 }
                                                       }
                                                           }
                                                 
    // 2 whites stops 3 blacks
    int nwmfghe = nwmf + nwmg + nwmh + nwme;
    int nbmfghe = nbmf + nbmg + nbmh + nbme;
    if ( ( nwmfghe == 2 ) && ( nbmfghe == 3 ) ){
    if (( b[22] == WHT_MAN ) && ( b[25] == WHT_MAN ))
    if ( ( b[12] == BLK_MAN ) && ( b[13] == BLK_MAN ) && ( b[17] == BLK_MAN ) )
    endgame -= 24;
    if (( b[31] == WHT_MAN ) && ( b[34] == WHT_MAN )){
    if ( ( b[26] == BLK_MAN ) && ( b[21] == BLK_MAN ) && ( b[22] == BLK_MAN ) )
    	endgame -= 24;                                                                                              
    	                                    }
                                                      }

    const S8 cscore_center[4][4] = {
    	                                  0 , -8,-20,-30,       // 0 versus 0,1,2,3
    	                                  8,   0,   -8, -20,        // 1 versus 0,1,2,3
    	                                  20,  8,    0,  -4,           // 2 versus  0,1,2,3
    	                                  30, 20,  4,   0           // 3 versus  0,1,2,3
    	                                        };
    	                                        
     const S8 cscore_edge[4][4] = {
    	                                  0 , -8,-10,-12,       // 0 versus 0,1,2,3
    	                                  8,   0,   -4, -6,        // 1 versus 0,1,2,3
    	                                  10,  4,    0,  - 2,           // 2 versus  0,1,2,3
    	                                  12,  6,    2,   0           // 3 versus  0,1,2,3
    	                                        };
    	 int nbmab = nbma + nbmb;
      int nbmcd = nbmc + nbmd;
      int nbmgh = nbmg + nbmh;
    	 int nbmef = nbme + nbmf;
  
    	 
      if ( nbmab > 3 ) nbmab = 3;
      if ( nbmcd > 3 ) nbmcd = 3;
    	 if ( nbmef > 3 ) nbmef = 3;
    	 if ( nbmgh > 3 ) nbmgh = 3;
    	 
    	 int nwmab = nwma + nwmb;
    	 int nwmcd = nwmc + nwmd;
    	 int nwmef = nwme + nwmf;
    	 int nwmgh = nwmg + nwmh;
    	 
    	 if ( nwmab > 3 ) nwmab = 3;
      if ( nwmcd > 3 ) nwmcd = 3;
    	 if ( nwmef > 3 ) nwmef = 3;
    	 if ( nwmgh > 3 ) nwmgh = 3;
    	 	
      eval += cscore_edge[nbmab][nwmab];
      eval += cscore_edge[nbmgh][nwmgh];
      eval += cscore_center[nbmcd][nwmcd];
      eval += cscore_center[nbmef][nwmef];
      
     index = -8*nbma - 4*nbmb -2*nbmc -1*nbmd + 1*nbme + 2*nbmf + 4*nbmg + 8*nbmh;
     balance -= abs(index);
     index = -8*nwma - 4*nwmb -2*nwmc - 1*nwmd  + 1*nwme + 2*nwmf + 4*nwmg + 8*nwmh;
     balance += abs(index);
     eval += balance;
     } // balance
              // mobility check
   int b_free = 0; // black's free moves counter
   int b_exchanges = 0; // black's exchanges counter
   int b_losing = 0; // black's apparently losing moves counter
   
   static U8 bonus[25] = {0,6,12,18,24,30,36,42,48,54,60,64,70,76,82,88,94,100,100,100,100,100,100,100,100};
   for ( i = 1;i <= num_bpieces;i++){
          if ( ( square = p_list[BLACK][i] ) == 0 ) continue;
          if ( b[square+5] == FREE ){
          do{
          int is_square_safe = 1;
          int can_recapture = 0;
          if ( ( b[square+10] & WHITE ) != 0 ){ // (1) danger
          is_square_safe = 0;
          // can white capture from square
          if ( ( ( b[square-4] & BLACK ) != 0 ) && ( b[square-8] == FREE ) ){b_losing++;break;}
          if ( ( ( b[square-5] & BLACK ) != 0 ) && ( b[square-10] == FREE ) ){b_losing++;break;}
          if ( ( ( b[square+4] & BLACK ) != 0 ) && ( b[square+8] == FREE ) ){b_losing++;break;}
          // can black recapture square
          if ( ( b[square-5] & BLACK ) != 0 )
          can_recapture = 1;          
          else
          if ( ( ( b[square-4] & BLACK ) != 0 )  && ( b[square+4] == FREE ) )
          can_recapture = 1;
          else
          if ( ( b[square-4] == FREE ) && ( ( b[square+4] & BLACK ) != 0 ) )
          can_recapture = 1;
          else{
          b_losing++;break;
                }
                                                                 } // (1) danger

          if ( ( ( b[square+9] & WHITE ) != 0 ) && ( b[square+1] == FREE ) ){ // (2) danger
          is_square_safe = 0;                         
          // can white capture from (square+1)
          if ( ( ( b[square-3] & BLACK ) != 0 ) && ( b[square-7] == FREE ) ) {b_losing++;break;}
          if ( ( ( b[square-4] & BLACK ) != 0 ) && ( b[square-9] == FREE ) ) {b_losing++;break;}
          if ( ( ( b[square+6] & BLACK ) != 0 ) && ( b[square+11] == FREE ) ) {b_losing++;break;}
          // can black recapture (square+1)
          if ( ( b[square-3] & BLACK ) != 0 )
          can_recapture = 1;          
          else
          if ( ( ( b[square-4] & BLACK ) != 0 )  && ( b[square+6] == FREE ) )
          can_recapture = 1;
          else
          if ( ( b[square-4] == FREE ) && ( ( b[square+6] & BLACK ) != 0 ) )
          can_recapture = 1;
          else{
          b_losing++;break;
                }
                                                                                                                      } // (2) danger
                                                          
          if ( ( ( b[square+4] & BLACK ) != 0 ) && ( ( b[square+8] & WHITE ) != 0 ) ){ // (3) danger
          is_square_safe = 0;
          // can white capture from square
          if ( b[square+10] == FREE ){b_losing++;break;}
          if ( ( ( b[square-5] & BLACK ) != 0 ) && ( b[square-10] == FREE ) ){b_losing++;break;}
          if ( ( ( b[square-4] & BLACK ) != 0 ) && ( b[square-8] == FREE ) ) {b_losing++;break;}
          // can black recapture square
          if ( b[square-5] == FREE )
          can_recapture = 1;
          else
          if ( ( b[square-4] & BLACK ) != 0 )
          can_recapture = 1;
          else{
          b_losing++;break;
                }
                                                                                                                                  } // (3) danger
                                                                                                                                  
          if ( ( b[square+9] == FREE ) && ( ( b[square+1] & WHITE ) != 0 ) ){ // (4) danger
          is_square_safe = 0;          	
          // can white capture from square+9
          if ( ( ( b[square+4] & BLACK ) != 0 ) && ( b[square-1] == FREE ) ){b_losing++;break;}
          if ( ( ( b[square+14] & BLACK ) != 0 ) && ( b[square+19] == FREE ) ){b_losing++;break;}
          if ( ( ( b[square+13] & BLACK ) != 0 ) && ( b[square+17] == FREE ) ){b_losing++;break;}
          // can black recapture square+9
          if ( ( b[square+13] & BLACK ) != 0 )
          can_recapture = 1;
          else
          if ( ( ( b[square+4] & BLACK ) != 0 )  && ( b[square+14] == FREE ) )
          can_recapture = 1;
          else
    	      if ( ( b[square+4] == FREE ) && ( ( b[square+14] & BLACK ) != 0 ) )
           can_recapture = 1;       
           else{
           b_losing++;break;
                  }   
                                                                                                                    } // (4) danger
                                                                                                                    
          // incomplete dangers
          if ( ( ( b[square-5] & BLACK ) != 0 ) && ( ( b[square-10] & WHITE ) != 0 ) ){ break; } // (5)
          if ( ( ( b[square-4] & BLACK ) != 0 ) && ( ( b[square-8] & WHITE ) != 0 ) ){ break; } // (6)
          // assert( is_square_safe^can_recapture == 1 );
          b_free += is_square_safe;
          b_exchanges += can_recapture;    
                 }while (0);
                                              };
                                                      
          if ( b[square+4] == FREE ){
          do{
          int is_square_safe = 1;
          int can_recapture = 0;
          if ( ( b[square+8] & WHITE ) != 0 ){ // (1) danger
          is_square_safe = 0;
          // can white capture from square
          if ( ( ( b[square-4] & BLACK ) != 0 ) && ( b[square-8] == FREE ) ){b_losing++; break;}
          if ( ( ( b[square+5] & BLACK ) != 0 ) && ( b[square+10] == FREE ) ){b_losing++; break;}
          if ( ( ( b[square-5] & BLACK ) != 0 ) && ( b[square-10] == FREE ) ){b_losing++; break;}
          // can black recapture square
          if ( ( b[square-4] & BLACK ) != 0 )
          can_recapture = 1;          
          else
          if ( ( ( b[square-5] & BLACK ) != 0 )  && ( b[square+5] == FREE ) )
          can_recapture = 1;
          else
          if ( ( b[square-5] == FREE ) && ( ( b[square+5] & BLACK ) != 0 ) )
          can_recapture = 1;
          else{
          b_losing++;break;
                }
                                                                } // (1) danger

          if ( ( ( b[square+9] & WHITE ) != 0 ) && ( b[square-1] == FREE ) ){ // (2) danger
          is_square_safe = 0;                         
          // can white capture from (square-1)
          if ( ( ( b[square-5] & BLACK ) != 0 ) && ( b[square-9] == FREE ) ){b_losing++; break;}
          if ( ( ( b[square-6] & BLACK ) != 0 ) && ( b[square-11] == FREE ) ){b_losing++; break;}
          if ( ( ( b[square+3] & BLACK ) != 0 ) && ( b[square+7] == FREE ) ){b_losing++; break;}
          // can black recapture (square-1)
          if ( ( b[square-6] & BLACK ) != 0 )
          can_recapture = 1;          
          else
          if ( ( ( b[square-5] & BLACK ) != 0 )  && ( b[square+3] == FREE ) )
          can_recapture = 1;
          else
          if ( ( b[square-5] == FREE ) && ( ( b[square+3] & BLACK ) != 0 ) )
          can_recapture = 1;
          else{
          b_losing++;break;
                }
                                                                                                                   } // (2) danger
                                                          
          if ( ( ( b[square+5] & BLACK ) != 0 ) && ( ( b[square+10] & WHITE ) != 0 ) ){ // (3) danger
          is_square_safe = 0;
          // can white capture from square
          if ( b[square+8] == FREE ) {b_losing++;break;}
          if ( ( ( b[square-5] & BLACK ) != 0 ) && ( b[square-10] == FREE ) ) {b_losing++;break;}
          if ( ( ( b[square-4] & BLACK ) != 0 ) && ( b[square-8] == FREE ) ){b_losing++;break;}
          // can black recapture square
          if ( b[square-4] == FREE )
          can_recapture = 1;
          else
          if ( ( b[square-5] & BLACK ) != 0 )
          can_recapture = 1;
          else{
          b_losing++;break;
                }
                                                                                                        } // (3) danger
                                                                                                        
          if ( ( b[square+9] == FREE ) && ( ( b[square-1] & WHITE ) != 0 ) ){  // (4) danger
          is_square_safe = 0;
          // can white capture from square+9
          if ( ( ( b[square+5] & BLACK ) != 0 ) && ( b[square+1] == FREE ) ){b_losing++;break;}
          if ( ( ( b[square+14] & BLACK ) != 0 ) && ( b[square+19] == FREE ) ){b_losing++;break;}
          if ( ( ( b[square+13] & BLACK ) != 0 ) && ( b[square+17] == FREE ) ){b_losing++;break;}
          // can black recapture square+9
          if ( ( b[square+14] & BLACK ) != 0 )
          can_recapture = 1;
          else
          if ( ( ( b[square+5] & BLACK ) != 0 )  && ( b[square+13] == FREE ) )
          can_recapture = 1;
          else
    	      if ( ( b[square+5] == FREE ) && ( ( b[square+13] & BLACK ) != 0 ) )
           can_recapture = 1;       
           else{
           b_losing++;break;
                  }
                              }
         // incomplete dangers
         if ( ( ( b[square-4] & BLACK ) != 0 ) && ( ( b[square-8] & WHITE ) != 0 ) ){ break;} // (5)
         if ( ( ( b[square-5] & BLACK ) != 0 ) && ( ( b[square-10] & WHITE ) != 0 ) ){ break;} // (6)
          // assert( is_square_safe^can_recapture == 1 );
          b_free += is_square_safe;
          b_exchanges += can_recapture;          
            }while (0);
                                };
                       } // for
                       
           int w_free = 0; // white's free moves counter
           int w_exchanges = 0; // white's exchanges counter
           int w_losing = 0; // whites's apparently losing moves counter
   
           for ( i = 1;i <= num_wpieces;i++){
           if ( ( square = p_list[WHITE][i] ) == 0 ) continue;
           if ( b[square-5] == FREE ){
            do{
           int is_square_safe = 1;
           int can_recapture = 0;
           if ( ( b[square-10] & BLACK ) != 0 ){ // (1) danger
           is_square_safe = 0;
           // can black capture from square
           if ( ( ( b[square+5] & WHITE ) != 0 ) && ( b[square+10] == FREE ) ){w_losing++; break;}
           if ( ( ( b[square+4] & WHITE ) != 0 ) && ( b[square+8] == FREE ) ){w_losing++; break;}
           if ( ( ( b[square-4] & WHITE ) != 0 ) && ( b[square-8] == FREE ) ){w_losing++; break;}
           // can white recapture square
           if ( ( b[square+5] & WHITE ) != 0 )
           can_recapture = 1;
           else          	
           if ( ( ( b[square+4] & WHITE ) != 0 ) && ( b[square-4] == FREE ) )
           can_recapture = 1;
           else
           if ( ( b[square+4] == FREE ) && ( ( b[square-4] & WHITE ) != 0 ) )
           can_recapture = 1;
           else{
           w_losing++;break;
                 }
                                                                  } // (1) danger
                                                                      
           if ( ( b[square-9] & BLACK ) != 0 && ( b[square-1] == FREE ) ){ // (2) danger
           is_square_safe = 0;
           // can black capture from (square-1)
           if ( ( ( b[square+3] & WHITE ) != 0 ) && ( b[square+7] == FREE ) ){w_losing++; break;}
           if ( ( ( b[square+4] & WHITE ) != 0 ) && ( b[square+9] == FREE ) ){w_losing++; break;}
           if ( ( ( b[square-6] & WHITE ) != 0 ) && ( b[square-11] == FREE ) ){w_losing++; break;}
           // can white recapture (square-1)
           if ( ( b[square+3] & WHITE ) != 0 )
           can_recapture = 1;
           else
           if ( ( ( b[square-6] & WHITE ) != 0 ) && ( b[square+4] == FREE ) )
           can_recapture = 1;
           else
   	       if ( ( b[square-6] == FREE ) && ( ( b[square+4] & WHITE ) !=0 ) )
   	       can_recapture = 1;
   	       else{
   	       w_losing++;break;
                  }
                                                                                                                  } // (2) danger
                                                                                                                  
          if ( ( b[square-4] & WHITE ) != 0 && ( b[square-8] & BLACK ) != 0 ){ // (3) danger
          is_square_safe = 0;
          // can black capture from square
          if ( b[square-10] == FREE ){w_losing++; break;}
          if ( ( ( b[square+5] & WHITE ) != 0 ) && ( b[square+10] == FREE ) ){w_losing++; break;}
          if ( ( ( b[square+4] & WHITE ) != 0 ) && ( b[square+8] == FREE ) ){w_losing++; break;}
          // can white recapture square
          if ( b[square+5] == FREE )
          can_recapture = 1;
          else
          if ( ( b[square+4] & WHITE ) != 0 )
          can_recapture = 1;
          else{
          w_losing++;break;
                }
                                                                                                             } // (3) danger
                                                                                                             
           if ( ( b[square-9] == FREE ) && ( b[square-1] & BLACK ) != 0 ){ // (4) danger
           is_square_safe = 0;
           // can black capture from square-9
           if ( ( ( b[square-4] & WHITE ) != 0 ) && ( b[square+1] == FREE ) ){w_losing++;break;}
           if ( ( ( b[square-14] & WHITE ) != 0 ) && ( b[square-19] == FREE ) ){w_losing++;break;}
           if ( ( ( b[square-13] & WHITE ) != 0 ) && ( b[square-17] == FREE ) ){w_losing++;break;}
           // can white recapture square-9
           if ( ( b[square-13] & WHITE ) != 0 )
           can_recapture = 1;
           else
           if ( ( ( b[square-14] & WHITE ) != 0 ) && ( b[square-4] == FREE ) )
           can_recapture = 1;
           else
   	       if ( ( b[square-14] == FREE ) && ( ( b[square-4] & WHITE ) !=0 ) )
   	       can_recapture = 1;
   	       else{
   	       w_losing++;break;
                  }
                          } // (4)
              
          // incomplete                                                                                                                           
          if (( b[square+5] & WHITE)!=0 && ( b[square+10] & BLACK)!=0){ break;} // (5)
          if (( b[square+4] & WHITE)!=0 && ( b[square+8] & BLACK)!=0){ break;} // (6)
    
          // assert( is_square_safe^can_recapture == 1 );	
          w_free += is_square_safe;
          w_exchanges += can_recapture;
           }while (0);   
                };
      
          if ( b[square-4] == FREE ){
          do{
          int is_square_safe = 1;
          int can_recapture = 0;
          if ( ( b[square-8] & BLACK ) != 0 ){ // (1) danger
          is_square_safe = 0;
          // can black capture from square
          if ( ( ( b[square+4] & WHITE ) != 0 ) && ( b[square+8] == FREE ) ){w_losing++; break;}
          if ( ( ( b[square+5] & WHITE ) != 0 ) && ( b[square+10] == FREE ) ){w_losing++; break;}
          if ( ( ( b[square-5] & WHITE ) != 0 ) && ( b[square-10] == FREE ) ){w_losing++; break;}
          // can white recapture square
          if ( ( b[square+4] & WHITE ) != 0 )
          can_recapture = 1;
          else
          if ( ( ( b[square+5] & WHITE ) != 0 ) && ( b[square-5] == FREE ) )
          can_recapture = 1;
          else
          if ( ( b[square+5] == FREE ) && ( ( b[square-5] & WHITE ) != 0 ) )
          can_recapture = 1;
          else{
          w_losing++;break;
                }
                                                                } // (1) danger

         if ( ( b[square-9] & BLACK ) != 0 && ( b[square+1] == FREE ) ){ // (2) danger
         is_square_safe = 0;
         // can black capture from (square+1)
         if ( ( ( b[square+6] & WHITE ) != 0 ) && ( b[square+11] == FREE ) ){w_losing++; break;}
         if ( ( ( b[square+5] & WHITE ) != 0 ) && ( b[square+9] == FREE ) ){w_losing++;break;}
         if ( ( ( b[square-3] & WHITE ) != 0 ) && ( b[square-7] == FREE ) ){w_losing++; break;}
         // can white recapture (square+1)
         if ( ( ( b[square+6] & WHITE ) != 0 ) )
         can_recapture = 1;
         else
         if ( ( ( b[square-3] & WHITE ) != 0 ) && ( b[square+5] == FREE ) )
         can_recapture = 1;
         else
         if ( ( b[square-3] == FREE ) && ( ( b[square+5] & WHITE ) != 0 ) )
         can_recapture = 1;
         else{
         w_losing++;break;
               }
                                                                                                                } // (2) danger
                                                                                                                
          if ( ( b[square-5] & WHITE ) != 0 && ( b[square-10] & BLACK ) != 0 ){ // (3) danger
          is_square_safe = 0;         	
          // can black capture from square
          if ( b[square-8] == FREE ){w_losing++; break;}
          if ( ( ( b[square+5] & WHITE ) != 0 ) && ( b[square+10] == FREE ) ){w_losing++; break;}
          if ( ( ( b[square+4] & WHITE ) != 0 ) && ( b[square+8] == FREE ) ){w_losing++; break;}
          // can white recapture square
          if ( b[square+4] == FREE )
          can_recapture = 1;
          else
          if ( ( b[square+5] & WHITE ) != 0 )
          can_recapture = 1;
          else{
          w_losing++;break;
                }       	
                                                                                                                            } // (3) danger
                                                                                                                            
         if ( ( b[square-9] == FREE ) && ( ( b[square+1] & BLACK ) != 0 ) ){ // (4) danger
         is_square_safe = 0;
          // can black capture from square-9
          if ( ( ( b[square-14] & WHITE ) != 0 ) && ( b[square-19] == FREE ) ){w_losing++;break;}
          if ( ( ( b[square-13] & WHITE ) != 0 ) && ( b[square-17] == FREE ) ){w_losing++;break;}
          if ( ( ( b[square-5] & WHITE ) != 0 ) && ( b[square-1] == FREE ) ){w_losing++;break;}
          // can white recapture square-9
          if ( ( b[square-14] & WHITE ) != 0 )
          can_recapture = 1;
          else
          if ( ( ( b[square-13] & WHITE ) != 0 ) && ( b[square-5] == FREE ) )
          can_recapture = 1;
          else
          if ( ( b[square-13] == FREE ) && ( ( b[square-5] & WHITE ) !=0 ) )
          can_recapture = 1;
          else{
          w_losing++;break;
                  }         	
              } // (4)
         	
         // incomplete dangers
         if ( ( ( b[square+4] & WHITE) !=0 ) && ( ( b[square+8] & BLACK ) !=0 ) ){ break;} // (5)
         if ( ( ( b[square+5] & WHITE) !=0 ) && ( ( b[square+10] & BLACK ) !=0 ) ){ break;} // (6)
         // assert( is_square_safe^can_recapture == 1 );		
         w_free += is_square_safe;
         w_exchanges += can_recapture;
             }while(0);
               };
               
               
             } // for
 
         	if ( b_exchanges ){
             eval += 4*b_exchanges;
       		                             }

           	if ( w_exchanges ){
            	eval -= 4*w_exchanges;
                                            }
                        
             eval += w_losing;
             eval -= b_losing;
             // free moves bonuses
             eval += bonus[b_free];
             eval -= bonus[w_free];
             
             if ( b_free == 0 && b_exchanges == 0 )
             eval -= 36;
             if ( b_free == 0 && b_exchanges == 1 )
             eval -= 36;
             	
             if ( w_free == 0 && w_exchanges == 0 )
             eval += 36;
             if ( w_free == 0 && w_exchanges == 1 )
             eval += 36;
                  } // if ( (nbk == 0) && (nwk == 0) )
              
     // developed black's single corner
     if ( ( b[5] == FREE ) && ( b[10] == FREE ) ){
     opening += 20;
     if (( b[14] != WHT_MAN ) && ( b[15] != WHT_MAN ))
     endgame += 20;
              }
     // developed white's single corner
     if ( ( b[40] == FREE )  && ( b[35] == FREE ) ){
     opening -= 20;
     if (( b[30] != BLK_MAN ) && ( b[31] != BLK_MAN ))
     endgame -= 20;
              }
    // one pattern ( V. K. Adamovich , Genadij I. Xackevich , Viktor Litvinovich )
    if ( ( b[15] == BLK_MAN ) && ( b[30] == WHT_MAN ) ){
    if ( ( b[16] == BLK_MAN ) && ( b[21] == BLK_MAN ) ){
    if ( ( b[24] == WHT_MAN ) && ( b[29] == WHT_MAN ) ){
    if ( ( b[20] == FREE ) && ( b[25] == FREE ) ){
    if ( color == BLACK )
    if (( b[23] != WHT_MAN ) || ( b[19] != BLK_MAN ))
    	eval += 8;
    	if ( color == WHITE )
    if (( b[22] != BLK_MAN ) || ( b[26] != WHT_MAN ))
    	eval -= 8;
    	                                                                              }
    	                                                                            }
    	                                                                          }
    	                                                                        }
    // parallel checkers
    if (( b[8] == BLK_MAN ) && ( b[16] == BLK_MAN ))
    if ( b[12] + b[7] + b[20] == FREE )
    	endgame -= 24;
    if (( b[13] == BLK_MAN ) && ( b[21] == BLK_MAN ))
    if ( b[12] + b[17] + b[25] == FREE )
    	endgame -= 24;
    if (( b[37] == WHT_MAN ) && ( b[29] == WHT_MAN ))
    if ( b[38] + b[33] + b[25] == FREE )
    	endgame += 24;
    if (( b[32] == WHT_MAN ) && ( b[24] == WHT_MAN ))
    if ( b[33] + b[28] + b[20] == FREE )
    	endgame += 24; 
   // passers on b6,d6,f6,h6
   if ( b[28] == BLK_MAN ){ // b6 ?
   do{
   if ( ( b[32] == FREE ) && ( b[37] == FREE ) ){ endgame += 24;break;}
   if ( color != BLACK ) break;
   if (( b[38] & WHITE ) != 0 ) break;
   if ( b[33] != FREE ) break;
   if ( ( ( b[37] & WHITE ) != 0 ) && ( b[29] == FREE )) break;
   if ( ( ( b[29] & WHITE ) != 0 ) && ( b[37] == FREE )) break;
   endgame += 12;
      }while(0);
                                           }
 
   if ( b[29] == BLK_MAN ){ // d6 ?
   do{
   if ( color != BLACK ) break;
   if ( b[34] != FREE ) break;
   if ( ( b[39] & WHITE ) != 0 ) break;
   if ( ( b[38] == FREE ) && ( ( b[30] & WHITE ) != 0 ) ) break;
   if ( ( ( b[38] & WHITE ) != 0 ) && ( b[30] == FREE ) ) break;	
   endgame += 12;
       }while(0);
   do{
   if ( color != BLACK ) break;
   if ( b[33] != FREE ) break;
   if ( ( b[37] & WHITE ) != 0 ) break;
   if ( ( b[38] == FREE ) && ( ( b[28] & WHITE ) != 0 ) ) break;
   if ( ( ( b[38] & WHITE ) != 0 ) && ( b[28] == FREE ) ) break;	
   endgame += 12;
       }while(0);
                                           }
                                          
   if ( b[30] == BLK_MAN ){ // f6 ?
   do{
   if ( color != BLACK ) break;
   if ( b[35] != FREE ) break;
   if ( ( b[40] & WHITE ) != 0 ) break;
   if ( ( b[39] == FREE ) && ( ( b[31] & WHITE ) != 0 ) ) break;
   if ( ( ( b[39] & WHITE ) != 0 ) && ( b[31] == FREE ) ) break;
   endgame += 12;
      }while(0);
   do{
   if ( color != BLACK ) break;
   if ( b[34] != FREE ) break;
   if ( ( b[38] & WHITE ) != 0 ) break;
   if ( ( b[39] == FREE ) && ( ( b[29] & WHITE ) != 0 ) ) break;
   if ( ( ( b[39] & WHITE ) != 0 ) && ( b[29] == FREE ) ) break;
   endgame += 12;
       }while(0);
                                            }
                                            
   if ( b[31] == BLK_MAN ){ // h6 ?
   if ( is_wht_kng(b) == 0 ){
   if ( b[39] == FREE && b[40] == WHT_MAN )
   endgame += 8;
   if ( b[24] == BLK_MAN )  // h6 + c5
   endgame += 8;
   do{
   if ( color != BLACK ) break;
   if (( b[39] & WHITE ) != 0 ) break;
   if ( b[35] != FREE ) break;
   if ( ( ( b[30] & WHITE ) != 0 ) && ( b[40] == FREE ) ) break;
   if ( ( b[30] == FREE ) && ( ( b[40] & WHITE ) != 0 ) ) break;
   endgame += 12;
      }while(0);
                                           }
                 }                                            
   // passers on a3,c3,e3,g3
   if ( b[14] == WHT_MAN ){ // a3 ?
   if ( is_blk_kng(b) == 0 ){
   if ( b[6] == FREE && b[5] == BLK_MAN )
   endgame -= 8;
   if ( b[21] == WHT_MAN ) // a3 + f4
   endgame -= 8;
   do{
   if ( color != WHITE ) break;
   if (( b[6] & BLACK) != 0) break;
   if ( b[10] != FREE ) break;
   if ( ( ( b[5] & BLACK ) != 0 ) && ( b[15] == FREE ) ) break;
   if ( ( b[5] == FREE ) && ( ( b[15] & BLACK) != 0 ) ) break;
   endgame -= 12;
      }while(0);
                                             }
                                            }

   if ( b[15] == WHT_MAN ){ // c3 ?
   do{
   if ( color != WHITE ) break;
   if ( b[10] != FREE ) break;
   if ( ( b[5] & BLACK ) != 0 ) break;
   if ( ( b[6] == FREE ) && ( ( b[14] & BLACK ) != 0 ) ) break;
   if ( ( ( b[6] & BLACK ) != 0 ) && ( b[14] == FREE ) ) break;
   endgame -= 12;
      }while(0);
   do{
   if ( color != WHITE ) break;
   if ( b[11] != FREE ) break;
   if ( ( b[7] & BLACK ) != 0 ) break;
   if ( ( b[6] == FREE ) && ( ( b[16] & BLACK ) != 0 ) ) break;
   if ( ( ( b[6] & BLACK ) != 0 ) && ( b[16] == FREE ) ) break;
   endgame -= 12;
       }while(0);
                                            }
                                            
   if ( b[16] == WHT_MAN ){ // e3 ?
   do{
   if ( color != WHITE ) break;
   if ( b[11] != FREE ) break;
   if ( ( b[6] & BLACK ) != 0 ) break;
   if ( ( b[7] == FREE ) && ( ( b[15] & BLACK ) != 0 ) ) break;
   if ( ( ( b[7] & BLACK ) != 0 ) && ( b[15] == FREE ) ) break;
   endgame -= 12;
      }while(0);
   do{
   if ( color != WHITE ) break;
   if ( b[12] != FREE ) break;
   if ( ( b[8] & BLACK ) != 0 ) break;
   if ( ( b[7] == FREE ) && ( ( b[17] & BLACK ) != 0 ) ) break;
   if ( ( ( b[7] & BLACK ) != 0 ) && ( b[17] == FREE ) ) break;
   endgame -= 12;
       }while(0);
                                            }

   if ( b[17] == WHT_MAN ){ // g3 ?
   do{
   if ( ( b[8] == FREE ) && ( b[13] == FREE ) ){ endgame -= 24;break;}
   if ( color != WHITE ) break;
   if ( (b[7] & BLACK) != 0 ) break;
   if ( b[12] != FREE ) break;   	
   if ( ( ( b[8] & BLACK ) != 0 ) && ( b[16] == FREE ) ) break;
   if ( ( b[8] == FREE ) && ( ( b[16] & BLACK ) != 0 ) ) break;
   endgame -= 12;
      }while(0);
                                              }
   // stroennost shashek  
   const int shadow = 5; // bonus for stroennost
   // stroennost for black
   if ( (b[16] & BLACK) != 0 )
   if ( (b[11] & BLACK) != 0 )
   if ( (b[6] & BLACK) != 0 )
   if ( b[21] == FREE )
   	eval += shadow;
   if ( (b[16] & BLACK) != 0 )
   if ( (b[12] & BLACK) != 0 )
   if ( (b[8] & BLACK) != 0 )
   if ( b[20] == FREE )
   	eval += shadow;
   if ( (b[20] & BLACK) != 0 )
   if ( (b[15] & BLACK) != 0 )
   if ( (b[10] & BLACK) != 0 )
   if ( b[25] == FREE )
   	eval += shadow;
   if ( (b[20] & BLACK) != 0 )
   if ( (b[16] & BLACK) != 0 )
   if ( (b[12] & BLACK) != 0 )
   if ( b[24] == FREE )
   	eval += shadow;
   if ( (b[25] & BLACK) != 0 )
   if ( (b[20] & BLACK) != 0 )
   if ( (b[15] & BLACK) != 0 )
   if ( b[30] == FREE )
   	eval += shadow;

   // stroennost for white
   if ( (b[29] & WHITE) != 0 )
   if ( (b[34] & WHITE) != 0 )
   if ( (b[39] & WHITE) != 0 )
   if ( b[24] == FREE )
   	eval -= shadow;
   if ( (b[29] & WHITE) != 0 )
   if ( (b[33] & WHITE) != 0 )
   if ( (b[37] & WHITE) != 0 )
   if ( b[25] == FREE )
   	eval -= shadow;
   if ( (b[25] & WHITE) != 0 )
   if ( (b[30] & WHITE) != 0 )
   if ( (b[35] & WHITE) != 0 )
   if ( b[20] == FREE )
   	eval -= shadow;
   if ( (b[25] & WHITE) != 0 )
   if ( (b[29] & WHITE) != 0 )
   if ( (b[33] & WHITE) != 0 )
   if ( b[21] == FREE )
   	eval -= shadow;
   if ( (b[20] & WHITE) != 0 )
   if ( (b[25] & WHITE) != 0 )
   if ( (b[30] & WHITE) != 0 )
   if ( b[15] == FREE )
   	eval -= shadow;
   // end stroennost
   int attackers,defenders;
   if ( b[24] == BLK_MAN ){ // b[24] safety
   	if ( b[25] == WHT_MAN )
    eval -= 10;
    if (( b[31] != BLK_MAN ) && ( b[34] == WHT_MAN ) && ( b[39] == WHT_MAN ))
   	eval -= 10; // bad for b[24]
    if ( ( b[23] == WHT_MAN ) && ( b[14] != FREE ) && ( b[15] == FREE ) && ( b[19] == FREE ))   	
   	eval -= 10; // bad for b[24]
   	attackers = defenders = 0;
    if ( b[5] == BLK_MAN )
    	   defenders++;
    if ( b[6] == BLK_MAN )
    	    defenders++;
    if ( b[10] == BLK_MAN )
    	   defenders++;
    if ( b[14] == BLK_MAN )
    	   defenders++;
    if ( b[29] == WHT_MAN )
    	   attackers++;
    if ( b[33] == WHT_MAN )
    	    attackers++;
    if ( b[37] == WHT_MAN )
    	   attackers++;
    if ( b[38] == WHT_MAN )
    	   attackers++;	   
    // must be defenders >= attackers
    if ( defenders < attackers )
    	  eval -= 20; 
                            }
   	                                        
   if ( b[21] == WHT_MAN ){ // b[21] safety
   	if ( b[20] == BLK_MAN )
   	  eval += 10;
    if ( ( b[14] != WHT_MAN ) && ( b[6] == BLK_MAN ) && ( b[11] == BLK_MAN ))
   	eval += 10; // bad for b[21]
   if ( ( b[22] == BLK_MAN ) && ( b[31] != FREE ) && ( b[30] == FREE ) && ( b[26] == FREE )) 
   	eval += 10; // bad for b[21]
   	attackers = defenders = 0;
    if ( b[39] == WHT_MAN )
         defenders++;
    if ( b[40] == WHT_MAN )
    	    defenders++;
    if ( b[35] == WHT_MAN )
    	    defenders++;
    if ( b[31] == WHT_MAN )
    	    defenders++;
    if ( b[16] == BLK_MAN )
    	    attackers++;
    if ( b[12] == BLK_MAN )
    	    attackers++;
    if ( b[8] == BLK_MAN )
    	    attackers++;
    if ( b[7] == BLK_MAN )
    	    attackers++;
    // must be defenders >= attackers
    if ( defenders < attackers )
    	  eval += 20;
   	                                          }                                     
   	                                        
   // blocked pieces in quadrants
   if ( ( b[23] == WHT_MAN ) && ( b[14] == BLK_MAN ) && ( b[15] == BLK_MAN ) && ( b[19] == BLK_MAN)){
   eval -= 40;
            }

   if (( b[11] == BLK_MAN ) && ( b[15] == BLK_MAN ) && ( b[16] == BLK_MAN ) && ( b[20] == BLK_MAN)){
   if (( b[24] == WHT_MAN ) && ( b[28] == WHT_MAN ) && ( b[25] == WHT_MAN ) && ( b[30] == WHT_MAN)){
   eval -= 40;
   if (( b[6] == BLK_MAN ) && ( b[10] == FREE ) && ( b[14] != WHT_MAN ))
   eval += 10;
   if (( b[7] == BLK_MAN ) && ( b[12] == FREE ) && ( b[17] != WHT_MAN ))
   eval += 10;
              }
            }
 
   if (( b[12] == BLK_MAN ) && ( b[16] == BLK_MAN ) && ( b[17] == BLK_MAN ) && ( b[21] == BLK_MAN)){
   if (( b[22] == WHT_MAN ) && ( b[26] == WHT_MAN ) && ( b[31] == WHT_MAN )){
   eval -= 40;
   if ( b[23] == BLK_MAN )
   eval += 5;
   if (( b[8] == BLK_MAN ) && ( b[13] == FREE ))
   eval += 5;
   	    }
   	  }
   
   if (( b[15] == BLK_MAN ) && ( b[19] == BLK_MAN ) && ( b[20] == BLK_MAN ) && ( b[24] == BLK_MAN)){
   eval -= 40;
   if (( b[10] == BLK_MAN ) && ( b[14] == FREE ))
   eval += 10;
   if (( b[11] == BLK_MAN ) && ( b[16] == FREE ) && ( b[21] != WHT_MAN ))
   eval += 10;
            }
   
   if (( b[16] == BLK_MAN ) && ( b[20] == BLK_MAN ) && ( b[21] == BLK_MAN ) && ( b[25] == BLK_MAN)){
   eval -= 40;
   if (( b[11] == BLK_MAN ) && ( b[15] == FREE ) && ( b[19] != WHT_MAN ))
   eval += 10;
   if (( b[12] == BLK_MAN ) && ( b[17] == FREE ) && ( b[22] != WHT_MAN ))
   eval += 10;
                                      }
   	
   if ( ( b[34] == WHT_MAN ) && ( b[31] == WHT_MAN ) && ( b[17] == BLK_MAN ) && ( b[21] == BLK_MAN ) && ( b[22] == BLK_MAN ) && ( b[26] == BLK_MAN)){
   eval -= 40;
               }
   //*********************************** for white color
   if ( ( b[22] == BLK_MAN ) && ( b[30] == WHT_MAN ) && ( b[31] == WHT_MAN ) && (b[26] == WHT_MAN)){
   eval += 40;
                    }
                    
   if (( b[33] == WHT_MAN ) && ( b[28] == WHT_MAN ) && ( b[29] == WHT_MAN ) && (b[24] == WHT_MAN)){
   if (( b[23] == BLK_MAN ) && ( b[19] == BLK_MAN ) && ( b[14] == BLK_MAN )){
   eval += 40;
   if ( b[22] == WHT_MAN )
   eval -= 5;
   if (( b[37] == WHT_MAN ) && ( b[32] == FREE ))
   eval -= 5;
                }
              }
 
   if (( b[34] == WHT_MAN ) && ( b[29] == WHT_MAN ) && ( b[30] == WHT_MAN ) && (b[25] == WHT_MAN)){
   if (( b[15] == BLK_MAN ) && ( b[20] == BLK_MAN ) && ( b[21] == BLK_MAN ) && ( b[17] == BLK_MAN)){
   eval += 40;
   if (( b[38] == WHT_MAN ) && ( b[33] == FREE ) && ( b[28] != BLK_MAN  ))
   eval -= 10;
   if (( b[39] == WHT_MAN ) && ( b[35] == FREE ) && ( b[31] != BLK_MAN  ))
   eval -= 10;
                  }
                }
   if ( ( b[11] == BLK_MAN ) && ( b[14] == BLK_MAN ) && ( b[28] == WHT_MAN ) && ( b[23] == WHT_MAN ) && ( b[24] == WHT_MAN ) && (b[19] == WHT_MAN)){
   eval += 40;
         }
   
   if (( b[29] == WHT_MAN ) && ( b[24] == WHT_MAN ) && ( b[25] == WHT_MAN ) && (b[20] == WHT_MAN)){
   eval += 40;
   if (( b[33] == WHT_MAN ) && ( b[28] == FREE ) && ( b[23] != BLK_MAN  ))
   eval -= 10;
   if (( b[34] == WHT_MAN ) && ( b[30] == FREE ) && ( b[26] != BLK_MAN  ))
   eval -= 10;
                            }   
   if (( b[30] == WHT_MAN ) && ( b[25] == WHT_MAN ) && ( b[26] == WHT_MAN ) && (b[21] == WHT_MAN)){
   eval += 40;
   if (( b[34] == WHT_MAN ) && ( b[29] == FREE ) && ( b[24] != BLK_MAN  ))
   eval -= 10;
   if (( b[35] == WHT_MAN ) && ( b[31] == FREE ))
   eval -= 10;
                        }
        
    // phase mix
    // smooth transition between game phases
    eval += ((opening * phase + endgame * antiphase )/24);
    eval &= ~(GrainSize - 1);
   // negamax formulation requires this:
   eval = ( color == BLACK ) ? eval : -eval;
   EvalHash[ (U32) ( HASH_KEY & EC_MASK ) ] = (HASH_KEY & 0xffffffffffff0000) | ( eval & 0xffff);
   return (eval);
                    }

static struct  coor numbertocoor(int n)
        {
    /* turns square number n into a coordinate for checkerboard */
  struct coor c;

   switch(n)
      {
      case 5:
         c.x=0;c.y=0;
         break;
      case 6:
         c.x=2;c.y=0;
         break;
      case 7:
         c.x=4;c.y=0;
         break;
      case 8:
         c.x=6;c.y=0;
         break;
      case 10:
         c.x=1;c.y=1;
         break;
      case 11:
         c.x=3;c.y=1;
         break;
      case 12:
         c.x=5;c.y=1;
         break;
      case 13:
         c.x=7;c.y=1;
         break;
      case 14:
         c.x=0;c.y=2;
         break;
      case 15:
         c.x=2;c.y=2;
         break;
      case 16:
         c.x=4;c.y=2;
         break;
      case 17:
         c.x=6;c.y=2;
         break;
      case 19:
         c.x=1;c.y=3;
         break;
      case 20:
         c.x=3;c.y=3;
         break;
      case 21:
         c.x=5;c.y=3;
         break;
      case 22:
         c.x=7;c.y=3;
         break;
      case 23:
         c.x=0;c.y=4;
         break;
      case 24:
         c.x=2;c.y=4;
         break;
      case 25:
         c.x=4;c.y=4;
         break;
      case 26:
         c.x=6;c.y=4;
         break;
      case 28:
         c.x=1;c.y=5;
         break;
      case 29:
         c.x=3;c.y=5;
         break;
      case 30:
         c.x=5;c.y=5;
         break;
      case 31:
         c.x=7;c.y=5;
         break;
      case 32:
         c.x=0;c.y=6;
         break;
      case 33:
         c.x=2;c.y=6;
         break;
      case 34:
         c.x=4;c.y=6;
         break;
      case 35:
         c.x=6;c.y=6;
         break;
      case 37:
         c.x=1;c.y=7;
         break;
      case 38:
         c.x=3;c.y=7;
         break;
      case 39:
         c.x=5;c.y=7;
         break;
      case 40:
         c.x=7;c.y=7;
         break;
      }
     return c;
   }


static void setbestmove( struct move2 move)
{
   int i;
   U8 from, to;
   int jumps;
   struct coor c1;

   jumps = move.l -2;

   from = FROM( &move);
   to = TO( &move);

   GCBmove.from = numbertocoor(from);
   GCBmove.to = numbertocoor(to);
   GCBmove.jumps = jumps;
   GCBmove.newpiece =  (unsigned)( (move.m[1]) >> 6 );
   GCBmove.oldpiece =  (unsigned)( (move.m[0]) >> 6 );


   for ( i = 2; i < move.l; i++ ){
              GCBmove.del[i-2] = numbertocoor( (unsigned)((move.m[i]) & 63 ));
              GCBmove.delpiece[i-2] = ( (unsigned)((move.m[i]) >> 6));
                                             }

  if ( jumps > 1 )
     {
         for ( i = 2; i < move.l; i++ )
                      {
                            c1 = numbertocoor( move.path[i - 1] );
                            GCBmove.path[i - 1] = c1;
                      }
     }
 else
  {
   GCBmove.path[1] = numbertocoor(to);
  }

}

static __inline U8  FROM( struct move2 *move ){
  return ( (move->m[0]) & 0x3f );
                        }
 
static __inline U8  TO( struct move2 *move ){
  return ( (move->m[1]) & 0x3f );
                        }

static void movetonotation(struct move2 move,char str[80])
{
   unsigned int j,from,to;
   char c;

   from = FROM( &move);
   to = TO( &move);
   from = from-(from/9);
   to = to-(to/9);
   from -= 5;
   to -= 5;
   j=from%4;from-=j;j=3-j;from+=j;
   j=to%4;to-=j;j=3-j;to+=j;
   from++;
   to++;
   c='-';
   if(move.l>2) c='x'; // capture or normal ?
   sprintf(str,"%2li%c%2li",from,c,to);
  }

U64 rand64(void){
     // Credits: JLKISS64 RNG from David Jones, UCL Bioinformatics Group
    	// seed variables
	static U64 x = 123456789123, y = 987654321987;
	static U32 z1 = 43219876, c1 = 6543217, z2 = 21987643, c2 = 1732654;

	x = ((U64)1490024343005336237 )* x + 123456789;
	y ^= y << 21; y ^= y >> 17; y ^= y << 30;	// do not set y=0

	U64 t;
	t = ((U64)4294584393 ) * z1 + c1; c1 = t >> 32; z1 = t;
	t = ((U64)4246477509) * z2 + c2; c2 = t >> 32; z2 = t;

	return x + y + z1 + ((U64)z2 << 32);	// return 64-bit result
}

static void Create_HashFunction(void){
   // fills ZobristNumbers array with big random numbers
   	if (ZobristInitialized) return;
	else ZobristInitialized = true;

   int p,q;
   //srand((unsigned int) time(NULL));
    for ( p=5; p<=40 ; p++ )
      for ( q=0; q <=15 ; q++ )
        ZobristNumbers[p][q] = 0;
   for ( p = 5; p <= 40 ; p++ ){
   	for( q = 0; q <= 15 ; q++ ){
        ZobristNumbers[p][q] = rand64();
                         }
                     }
   HashSTM = rand64(); // constant random number - side to move
}

static U64 Position_to_Hashnumber( int b[46] , int color )
{
  U64 CheckSum = 0;
  int cpos;

  for ( cpos=5; cpos<41; cpos++ ) {
    if  ( ( b[cpos] != OCCUPIED ) && ( b[cpos] != FREE ) )
     CheckSum ^= ZobristNumbers[cpos][b[cpos]];
                                                       }

     if ( color == BLACK )
          CheckSum ^= HashSTM;

  return (CheckSum);
}

static void update_hash(struct move2 *move){
    // update HASH_KEY incrementally
    
    unsigned int contents,square;
    
    HASH_KEY ^= HashSTM;
    
    square = (move->m[0]) & 0x3f;
    contents = ((move->m[0]) >> 6);
    HASH_KEY ^= ZobristNumbers[square][contents];
    
    square = (move->m[1]) & 0x3f;
    contents = ((move->m[1]) >> 6);
    HASH_KEY ^= ZobristNumbers[square][contents];
    // captured pieces are below:
    for(register int i=move->l-1;i>1;i--){
    square = ((move->m[i]) & 0x3f);
    contents = ((move->m[i]) >> 6);
    HASH_KEY ^= ZobristNumbers[square][contents];
                                                        }
 }

static void TTableInit( unsigned int hash_mb){
 //
 //
       int j;
       U32 size;
       U32 target;

       if ( ttable != NULL ){
       free(ttable);
       ttable = NULL;
                                        }

       target = hash_mb;
       target *= 1024 * 1024;

       j = sizeof(TEntry);
		
	   //char str[32];
	   //itoa(j, str, 10);
	   //MessageBox(0, str, "tentry size", 0);
       assert( j ==16 ); // must be 16 bytes
            
       for (size = 1;( (size != 0) && (size <= target)); size *= 2)
       ;
       size /= 2;
       assert( (size > 0) && (size <= target) );
       // allocate table
       size /= 16;
       assert( (size != 0) && ( (size & (size-1))) == 0 ); // power of 2
       ttable  = ( TEntry *) malloc(size*16);
       memset(ttable,0,size*16);
       MASK = size - 4;
    }

 static void EvalHashClear()
    {
    int c;
    for ( c = 0; c < EC_SIZE ; c++)
        EvalHash[c] = 0;
    }

static int rootsearch(int b[46], int alpha, int beta, int depth,int color,int search_type){
//
//
    int bestvalue;
    int value;
    int i;
    int bestindex = 0;
    U8 bestfrom;
    U8 bestto;
    static int n;
    static int capture;
    static struct move2 movelist[MAXMOVES];
    int root_scores[MAXMOVES]; // root moves scores
    int newdepth;
    U64 L_HASH_KEY;
    
    // for displaying search info
    char ci_str[32]; // current index
    char value_str[16]; // current best value
    char depth_str[16];  // current depth
    char seldepth_str[16]; // current selective depth
    char speed_str[16]; // current speed
    char cm_str[64]; // currently executed move
    
    char string1[255];
    char string2[255];
    char string3[255];
    
    double elapsed; // time variable

    if (*play) return(0);
    nodes++; // increase node count
    const int old_alpha = alpha;
    
    // total number of pieces on board 
    unsigned int Pieces = g_pieces[BLK_MAN]+g_pieces[BLK_KNG]+g_pieces[WHT_MAN]+g_pieces[WHT_KNG];
 
    if ( search_type == SearchShort ){
    capture = Test_Capture(b,color);
    if ( capture )
    n = Gen_Captures(b, movelist, color,capture);
    else
    n = Gen_Moves(b,movelist, color);
    if( n == 0 ) 
    return (-MATE);
                                                           }
                                                           
    EdRoot[color] = EdAccess::not_found;                                                 
    /* check for database use */
    if ( Pieces <= EdPieces ){             
    if (  (!EdNocaptures) || (capture == 0) ){
    EdRoot[color] = EdProbe(b,color);
                                           }
        }
        
    if (EdRoot[color] == EdAccess::win) EdRoot[color ^ CC] = EdAccess::lose;
    else if (EdRoot[color] == EdAccess::lose) EdRoot[color ^ CC] = EdAccess::win;
    else EdRoot[color ^ CC] = EdRoot[color];

    for( i = 0;i < n; i++)
       root_scores[i] = -INFINITY;

   L_HASH_KEY = RepNum[realdepth] = HASH_KEY; // save HASH_KEY
   char **pv_str = 0;
   if (!pv_str)
		{
			pv_str = new char *[2];
             pv_str[0] = new char[1024];
		}
        for ( i = 0;i <= 3;i++ ){
                  killersf1[i] = 0;
                  killerst1[i] = 0;
                  killersf2[i] = 0;
                  killerst2[i] = 0;
                                           }
                 
                     
   //MoveToStr(movelist[0],*pv_str); // best move found so far
   inSearch = false;
   bestvalue = -INFINITY; // preset bestvalue
   // loop through all the moves in the root move list
   for ( i = 0; i<n ; i++ ){
   	     domove(b,&movelist[i],color);
   	     inSearch = false;
   	     newdepth = depth - 1;
         /******************* recursion**************************************************/
        if ( bestvalue == -INFINITY || depth <= 2 )
        value = -PVSearch(b,newdepth,-beta,-alpha,color^CC);
        else{
        	if ( newdepth <= 3 )
        value = -LowDepth(b,newdepth,-alpha ,color^CC);
        else
        value = -Search(b,newdepth,-alpha,color^CC,NodeCut, false );
        if ( value > alpha ) // research
        value = -PVSearch(b,newdepth, -alpha - 1 ,-alpha,color^CC);
        if ( value > alpha ) // research
        value = -PVSearch(b,newdepth, -beta,-alpha,color^CC);
              }
         /********************************************************************************/
                   	
         undomove(b,&movelist[i],color);
         // restore HASH_KEY
         HASH_KEY = L_HASH_KEY;
         if ( *play ) return (0);
         if ( capture == 0 ) // penalty for repeatable move
         if ( abs(value) <= HASHMATE )
         if ( ( ( (movelist[i].m[0]) >> 6) & KING ) != 0 )
         if ( g_pieces[(color|MAN)] != 0 )
         value--;  // penalty for repeatable move
         	// update move scores
		if (value <= alpha)
		{
		 root_scores[i] = old_alpha;
		}
		else if (value >= beta)
		{
		 root_scores[i] = beta;
		}
		else
		{
          root_scores[i] = value;
          }
         
        if ( value > bestvalue ){
            bestvalue = value;
            bestindex = i;
                                             }
        if ( value > alpha && search_type == SearchNormal )
        alpha = value; 
        if ( value >= beta )
        break;
#ifdef KALLISTO
   if ( pfSearchInfo ){
   elapsed = (clock()-start)/(double)TICKS;
   sprintf(string1,"");
   MoveToStr(movelist[i],string1); // currently executed move
   sprintf(string3,"");
   sprintf(string3,"%2i/%2i  ",i+1,n); // current move number / number of moves
   strcat(string3,string1);
   sprintf(string2,"");
   MoveToStr(movelist[bestindex],string2); // best move found so far
   if (pfSearchInfo) pfSearchInfo(bestvalue, depth, elapsed > 0 ? int(nodes / elapsed / 1000) : 0, string2, string3);
                                }
   if ( pfSearchInfoEx ){ // ToSha do not supports pfSearchInfoEx
   itoa(bestvalue, value_str, 10);
   itoa(depth, depth_str, 10);
   strcat(depth_str,"/");
   itoa(g_seldepth,seldepth_str, 10);
   strcat(depth_str,seldepth_str);
   elapsed = (clock()-start)/(double)TICKS;
   int speed = elapsed > 0 ? int(nodes / elapsed / 1000) : 0;
   itoa(speed, speed_str, 10);
   strcat(speed_str," kNps");
   sprintf(ci_str,""); // current index string
   sprintf(ci_str,"%2i/%2i   ",i+1,n); // current move number / number of moves
   MoveToStr(movelist[i],cm_str); // currently executed move
   strcat(ci_str,cm_str); // index + current move
   MoveToStr(movelist[bestindex],*pv_str); // best move found so far
   pfSearchInfoEx(value_str, depth_str,speed_str,pv_str,ci_str);
                                  }
 
#endif  
              }; // end move loop in the root move list
         
         bestrootmove = movelist[bestindex];
        
#ifdef KALLISTO
          
           if (pfSearchInfoEx){
           if ( i == n ) i--;
           itoa(bestvalue, value_str, 10);
           itoa(depth, depth_str, 10);
           strcat(depth_str,"/");
           itoa(g_seldepth,seldepth_str, 10); // selective depth
           strcat(depth_str,seldepth_str); // depth / selective depth
           elapsed = (clock()-start)/(double)TICKS;
           int speed = elapsed > 0 ? int(nodes / elapsed / 1000) : 0;
           itoa(speed, speed_str, 10);
           strcat(speed_str," kNps");
           sprintf(ci_str,""); // current index string
           sprintf(ci_str,"%2i/%2i   ",i+1,n); // current move number / number of moves
           MoveToStr(movelist[i],cm_str); // currently executed move 
           strcat(ci_str,cm_str);
           MoveToStr(movelist[bestindex],*pv_str); // best move found so far
           pfSearchInfoEx(value_str, depth_str,speed_str,pv_str,ci_str);
                                       }
     
#endif
               
                  bestfrom = FROM( &bestrootmove );
                  bestto = TO( &bestrootmove );
                  // and save the position in the hashtable
                  int f = (bestvalue <= old_alpha ? UPPER : bestvalue >= beta ? LOWER : EXACT);
                  hashstore( value_to_tt( bestvalue),depth,bestfrom, bestto,f);
                      // and order the movelist with insert ( stable ) sort
	                 for (  i = 1; i < n; i++ )
                        {
		            int rvalue = root_scores[i];
                     move2 tmpmove = movelist[i];
                     int j = i - 1;
                     for ( ; j >= 0 && root_scores[j] < rvalue; j--)
                                		{
                           			root_scores[j + 1] =root_scores[j];
			                           movelist[j + 1] = movelist[j];
                                       	}
                          		root_scores[j + 1] = rvalue;
                                movelist[j + 1] = tmpmove;
                        }
    return bestvalue;
}


static int compute( int b[46],int color, int time, char output[256]){
   // compute searches the best move on position b in time time.
   // it uses iterative deepening to drive the PV_Search.
   int depth;
   int i;
   int value;
   int lastvalue;
   int newvalue;
   int dummy=0,alpha,beta;
   U8 bestfrom = 0;
   U8 bestto = 0;
   U8 bestfrom2 = 0;
   U8 bestto2 = 0;   
   int bestindex;
   int n;
   double t, elapsed=0; // time variables
   struct move2 movelist[MAXMOVES];
   int  ValueByIteration[MAXDEPTH+1] ;
   bool Problem = false;
   struct move2 lastbest;
   char str[256];
   char pv[256];
   nodes = 0;
   sprintf(output,"KestoG engine 1.5");
   init_piece_lists(b);
   n = Gen_Captures(b, movelist, color,1);
   if(!n)
   n = Gen_Moves(b,movelist, color);
   if ( n == 0 )
   return (-MATE);
   	
   searches_performed_in_game++; // increase generation
   
#ifdef KALLISTO
  if ( n == 1 ){ // only one move to do=>return this move instantly
   value = 0;
   bestrootmove=movelist[0];
   MoveToStr(bestrootmove,str);
   if (pfSearchInfo) pfSearchInfo(value,0, elapsed > 0 ? int(nodes / elapsed / 1000) : 0, str, 0);
   movetonotation(bestrootmove,str);
   sprintf(output,"[only move][depth %i][move %s][time %.2fs][eval %i][nodes %i]",0,str,elapsed,value,0);
   printf("\n%s",output);
   setbestmove(bestrootmove);
   domove2(b,&bestrootmove,color);
   return (0);
                   }
#endif
           
                 // History is not cleared,but scaled down between moves
                 // scaling down History
                 for ( int p = 0; p <= 1023; p++){
                 //history_tabular[p] = 0;
                 history_tabular[p] = ( history_tabular[p] ) / 2;
                                                                    }
                  // Clear the killer moves
                  for ( i = 0; i <= (MAXDEPTH + 1) ;i++ ){
                  killersf1[i] = 0;
                  killerst1[i] = 0;
                  killersf2[i] = 0;
                  killerst2[i] = 0;
                                                                     }
                     // clear repetition checker
                  memset(RepNum,0,MAXDEPTH*sizeof(U64));
                  maxtime = time;
                  HASH_KEY = Position_to_Hashnumber(b,color);
                  start = clock();
                  g_Panic = 2;
                  g_seldepth = 0;
                  realdepth = 0;
                  value = rootsearch(b,-INFINITY,INFINITY,2,color,SearchShort);
                  alpha = -INFINITY;
                  beta = INFINITY;
                  for ( depth = 3; depth < MAXDEPTH; depth += 2){
                                       g_Panic = Problem ? 3 : 2; // if Problem then add more time
                                       lastvalue = value;
                                       lastbest = bestrootmove;
                                       repeat_search:
                                       g_seldepth = 0;
                                       HASH_KEY = Position_to_Hashnumber(b,color);
                                       init_piece_lists(b);
                                       realdepth = 0;
                                        /* do a search with aspiration window */
                           
                                       value = rootsearch(b,alpha,beta,depth,color,SearchNormal);
	
    elapsed = (clock()-start)/(double)TICKS;
   // interrupt by user or time is up
   if (*play){
   value = lastvalue;
   bestrootmove = lastbest;
   depth-=2;
   movetonotation(bestrootmove,str);
   if ( nodes < 1048576 )
sprintf(output,"[done][depth %i][move %s][time %.2fs][eval %i][nodes %i]",depth,str,elapsed,value,nodes);
   else
sprintf(output,"[done][depth %i][move %s][time %.2fs][eval %i][nodes %.1fM]",depth,str,elapsed,value,(float)nodes/(1024*1024));
                              printf("\n%s",output);
                              setbestmove(bestrootmove);
                              domove2(b,&bestrootmove,color);
#ifdef KALLISTO
   char value_str[16];
   itoa(value, value_str, 10);
   char depth_str[16];
   itoa(depth, depth_str, 10);
   char seldepth_str[16];
   itoa(depth, depth_str, 10);
   strcat(depth_str,"/");
   itoa(g_seldepth,seldepth_str, 10);
   strcat(depth_str,seldepth_str);
   int speed = elapsed > 0 ? int(nodes / elapsed / 1000) : 0;
   char speed_str[16];
   itoa(speed, speed_str, 10);
   strcat(speed_str," kNps");
   char **pv_str = 0;
  if (!pv_str)
		{
             pv_str = new char *[2];
             pv_str[0] = new char[1024];
		}

   MoveToStr(bestrootmove,*pv_str);
   char current_move[16];
   sprintf(current_move,"N/A");
   if (pfSearchInfoEx) pfSearchInfoEx(value_str, depth_str,speed_str,pv_str,current_move);
   return value;
#endif
 
                                        } // *play
         /* check if aspiration window holds */

         if ( abs(value) < 10000 && depth >= 21 ){
         if (  value >= beta  ){
             		alpha = -INFINITY;
         		beta = INFINITY;
         		goto repeat_search;
                                          }
            if (  value <= alpha ){
             		alpha = -INFINITY;
         		beta = INFINITY;
         		goto repeat_search;
                                              }                              
                                          
           alpha = value - RADIUS;
           if (alpha <= -INFINITY) alpha = -INFINITY;
           beta = value + RADIUS;
           if (beta >= INFINITY ) beta = INFINITY;
                                                                               } 
       else{
       	   alpha = -INFINITY;
            beta = INFINITY;
              };
      ValueByIteration[depth] = value;
      Problem = ((depth>=11) && (value<=ValueByIteration[depth-2]-30) &&  (ValueByIteration[depth-2]<=ValueByIteration[depth-4]-40 ));
      bestfrom = bestto = 0;
      bestfrom2 = bestto2 = 0;
      bestindex = 127;
      // get best move from hashtable:
      // we always store in the last call to rootsearch, so there MUST be a move here!
      hashretrieve(&dummy,depth, &dummy, alpha, beta, &bestfrom, &bestto, &bestfrom2,&bestto2, &dummy);
    
    for ( i = 0; i < n; i++ ){
    if ( ( FROM( &movelist[i] ) == bestfrom) && (TO( &movelist[i] ) == bestto) ){
             bestindex = i;
             break;
                                }
                                      }
      assert( bestindex != 127 );
      assert( FROM( &bestrootmove) == FROM( &movelist[bestindex] ));
      assert( TO( &bestrootmove) == TO( &movelist[bestindex] ));
      movetonotation(movelist[bestindex],str);
      if ( nodes < 1048576 )
sprintf(output,"[thinking][depth %i][move %s][time %.2fs][eval %i][nodes %i]",depth,str,elapsed,value,nodes);
      else
sprintf(output,"[thinking][depth %i][move %s][time %.2fs][eval %i][nodes %.1fM]",depth,str,elapsed,value,(float)nodes/(1024*1024));
      printf("\n%s",output);
   
                      // break conditions:
                      // time elapsed
                      t = clock();
 /* don't bother going deeper if we've already used 75% of our time since we likely won't finish */
 if ( (t-start) > (0.75*maxtime) ) break;

      // found a win
 if ( depth >= 50 && value >= MATE-depth )
         break;
                        };  // end for iterative deepening loop
   movetonotation(movelist[bestindex],str);
   retrievepv(b,pv,color);
   if ( nodes < 1048576 )
 sprintf(output,"[done][depth %i][move %s][time %.2fs][eval %i][nodes %i][pv %s]",depth,str,elapsed,value,nodes,pv);
   else
 sprintf(output,"[done][depth %i][move %s][time %.2fs][eval %i][nodes %.1fM][pv %s]",depth,str,elapsed,value,(float)nodes/(1024*1024),pv);
   printf("\n%s",output);
   setbestmove(bestrootmove);
   domove2(b, &bestrootmove,color);
   return value;
   }

static int QSearch(int b[46],int alpha,int beta,int color){
	// quiescent search
	// expands captures and promotions
   
    /* time check */
    if ( !( ++nodes & 0x1fff ) )
    if ( ( (clock()-start)>maxtime*g_Panic ))
    (*play) = 1;
    
    /* return if calculation interrupt */
    if (*play) return (0);
  	
    /* stop search if maximal search depth is reached */
    if ( realdepth >= MAXDEPTH ) return evaluation(b,color,alpha,beta);
    unsigned capture;   
    int value;

    capture = Test_Capture(b,color);

  if ( capture == 0 ){

  if ( realdepth > g_seldepth ) g_seldepth = realdepth; // new selective depth
    	
  int best_value = evaluation(b,color,alpha,beta) + 2; // static value + turn
  if ( best_value < -HASHMATE ) return ( best_value - 2 );
  if( best_value > alpha ){
  alpha = best_value;
  if ( best_value >= beta ) // selective deviation
  return best_value;
                                         }

  unsigned int n;
  struct move pmovelist[7];
  n = Gen_Proms(b,pmovelist,color); // generate promotions
  if ( n == 0 ) return best_value; // no promotions => return evaluation
  U64 L_HASH_KEY = HASH_KEY;  // local variable  for saving position's HASH_KEY
  unsigned int i;
  // move loop over promotions
  for( i = 0; i < n; i++ ){
  doprom(b,&pmovelist[i],color);
  value = -QSearch(b, -beta, -alpha,color^CC);
  undoprom(b,&pmovelist[i],color);
  HASH_KEY = L_HASH_KEY;
  if ( value > best_value ){
  best_value = value;
  if ( value > alpha ){
  alpha = value;
  if ( value >= beta ) return value;
                                 }
                                        }
                              } // move loop
      return best_value;
       }
  else{
        int i,n;
        struct move2 movelist[MAXMOVES];
        n = Gen_Captures(b,movelist,color,capture);
        int SortVals[MAXMOVES];
                  // sort captures list
                  for ( i=0;i<n;i++ ){
                  if (is_promotion(&movelist[i]) == 1) SortVals[i] = 4000;
                  else
                  SortVals[i] = 0;
                  for ( int j = 2;j < movelist[i].l;j++ )
                  SortVals[i] += MVALUE[ (movelist[i].m[j]) >> 6 ];
                                               }
                
                  int maxvalue = -INFINITY;    // preset maxvalue
                  U64 L_HASH_KEY = HASH_KEY;  // local variable  for saving position's HASH_KEY
                  // move loop
                  while ( pick_next_move( &i,&SortVals[0], n ) != 0 ){
                  domove(b,&movelist[i],color);
                  value = -QSearch(b, -beta, -alpha, color^CC);
                  undomove(b,&movelist[i],color);
                  HASH_KEY = L_HASH_KEY;
                  if (value > maxvalue){
                  maxvalue = value;
                  if (value > alpha){
                  if (value >= beta) break;
                  alpha = value;
                                                }
                                                       }
                                                          } // move loop
            return maxvalue;
                    }
 }
 
                 /* Main search function */
                 /* Principal Variation Search,also known as Nega-Scout */
 static int PVSearch(int b[46],int depth,int alpha,int beta,int color){
                  /* mate distance pruning */
                  alpha = MAX(value_mated_in(realdepth), alpha);
                  beta = MIN(value_mate_in(realdepth+1), beta);
                  if (alpha >= beta)
                  return alpha;
        
                  // horizon ?
                  if ( depth <= 0 ) return QSearch(b,alpha,beta,color);
                  /* time check */
                   
                  if ( !( ++nodes & 0x1fff ) )
                  if ( ((clock()-start) > maxtime*g_Panic) )
                	(*play) = 1;

                  /* return if calculation interrupt */
                  if (*play) return (0);

                
                  /* stop search if maximal search depth is reached */
                 if ( realdepth >= MAXDEPTH ) return evaluation(b,color,alpha,beta);
                 
                 if ( realdepth > g_seldepth ) g_seldepth = realdepth; // new selective depth
                 const int oldalpha = alpha;
                  /* check for database use */
                 
  unsigned int Pieces=g_pieces[BLK_MAN]+g_pieces[BLK_KNG]+g_pieces[WHT_MAN]+g_pieces[WHT_KNG];
 
  if ( (!EdNocaptures) && (Pieces <= EdPieces) )
   {
      int res = EdProbe(b,color);
      if (res != EdAccess::not_found)
      {
         if ( res != EdRoot[color] || !Reversible[realdepth - 1] )
         {
      if (res == EdAccess::win)  return ED_WIN - 100*Pieces;
      if (res == EdAccess::lose) return -ED_WIN + 100*Pieces;
      if (res == EdAccess::draw) return (0);
      MessageBox(0, "unknown value from EdAccess", "Error", 0);
         }
      }
   }
        int value;
          
         U64 L_HASH_KEY = RepNum[realdepth] = HASH_KEY;  // local variable  for saving position's HASH_KEY
         // draw by triple repetition ?
        
         if ( g_pieces[WHT_KNG] && g_pieces[BLK_KNG] ){
   	    for (int i = 4; i <= realdepth; i += 2){
         if ( RepNum[ realdepth - i ] == HASH_KEY )
         return (0);
   		              }
                                                        }
  
   U8 bestfrom,bestto;
   U8 bestfrom2,bestto2;
   bestfrom =  bestto = 0; // best move's from-to squares
   bestfrom2 = bestto2 = 0;
   int tr_depth = 1024;
   int tr_depth2;
   
                  // hashlookup
   
                  TEntry *entry;
                  register unsigned int j;
                  U32 lock;
                  int move_depth = 0;                                 
                  //entry = ttable + (U32)(HASH_KEY & MASK);
                  entry = ttable + (HASH_KEY & MASK);
                  lock = (HASH_KEY >> 32); // use the high 32 bits as lock

                  for( j = 0; j < REHASH ; j++,entry++ ){
                  if ( (entry->m_lock) == lock){
                  	tr_depth2 = (entry->m_depth); // depth stored in TT
                       if ( tr_depth2 > move_depth ){
                       move_depth = tr_depth;
                       bestfrom = (entry->m_best_from);
                       bestto = (entry->m_best_to);
                       bestfrom2 = (entry->m_best_from2);
                       bestto2 = (entry->m_best_to2);
                       tr_depth = (entry->m_depth);
                                                                          }
                                                                    }
                                                                               } // for
   
   int i;
   struct move2 movelist[MAXMOVES];
   // check if the side to move has a capture
   int capture;
   capture = Test_Capture(b,color);   // is there a capture for the side to move ?
      
   if ( capture == 0 ){
      /* check for compressed ( without captures ) database use */
      if ( EdNocaptures && (Pieces <= EdPieces) ){
      int res = EdProbe(b,color);
      if (res != EdAccess::not_found)
      {
         if (res != EdRoot[color] || !Reversible[realdepth - 1])
         {
      if (res == EdAccess::win) return ED_WIN - 100*Pieces;
      if (res == EdAccess::lose) return -ED_WIN + 100*Pieces;
      if (res == EdAccess::draw) return (0);
      MessageBox(0, "unknown value from EdAccess", "Error", 0);
         }
      }
                                                                            }
                                 }
            
   int n;
   if ( capture ){
   n = Gen_Captures(b,movelist,color,capture);
                        }
   else{
   n = Gen_Moves(b,movelist,color);
   // if we have no moves and no captures :
   if ( n == 0 ){
   return (realdepth-MATE); // minus sign will be negated in negamax framework
                     };
         }

 /* enhanced transposition cutoffs: do every move and check if the resulting
 position is in the hashtable. if yes, check if the value there leads to a cutoff
 if yes, we don't have to search */
 
       if ( tr_depth < depth && depth >= ETCDEPTH && beta > -1500 ){
       U8 dummy;
       int dummy2;
       dummy = dummy2 = 0;
       int ETCvalue;
       for( i = 0; i < n; i++ ){
       // do move
       // domove(b,&movelist[i] );
       update_hash(&movelist[i]); // instead domove
       // do the ETC lookup:with reduced depth and changed color
       ETCvalue = -INFINITY;
       if (hashretrieve(&dummy2,depth-1,&ETCvalue,-beta,-alpha,&dummy,&dummy,&dummy,&dummy,&dummy2)){
       // if one of the values we find is > beta we quit!
       if ( (-ETCvalue) >= beta ){
       // before we quit: restore all stuff
       // undomove(b,&movelist[i] );
       HASH_KEY = L_HASH_KEY;
       hashstore(value_to_tt(-ETCvalue),depth,FROM( &movelist[i]), TO( &movelist[i]),LOWER);
       return (-ETCvalue);
                                                }
                                                       } // if hashretrieve
      // undomove(b,&movelist[i] );
       HASH_KEY = L_HASH_KEY;
                                     } // for
                                               } // ETC
                         int SortVals[MAXMOVES];
                         // sort move list
                         // fill SortVals array with move's values
                    
                         int hindex;
                         if ( capture == 0 ){
                         for ( i = 0; i < n; i++ ){ // loop over moves
                         if ( FROM( &movelist[i]) == bestfrom ){
                         if ( TO( &movelist[i]) == bestto ){
                         SortVals[i] = 1000000;continue;
                                                                      }
                                                                    }
                         if ( FROM( &movelist[i]) == bestfrom2 ){
                         if ( TO( &movelist[i]) == bestto2 ){
                         SortVals[i] = 1000000 - 2;continue;
                                                                      }
                                                                    }
                                                                                       
                        if ( is_promotion(&movelist[i]) == 1 ){
                        SortVals[i] = 600000;continue;
                                                                                     }
                         if ( FROM( &movelist[i]) == killersf1[realdepth] ){
                         if ( TO( &movelist[i]) == killerst1[realdepth] ){
                         SortVals[i] = 600000 - 2;continue;
                                                                      }
                                                                    }
                                                                       
                          if (FROM( &movelist[i]) == killersf2[realdepth] ){
	                     if ( TO( &movelist[i]) == killerst2[realdepth] ){
	                     SortVals[i] = 600000 - 4;continue;
                                                            }
                                                          }
                      
                        hindex = (SQUARE_TO_32( FROM( &movelist[i]) ) << 5 ) + SQUARE_TO_32(TO( &movelist[i]) );
                        SortVals[i] = history_tabular[hindex];
                         
                          } // loop over moves
                                     
                                    }
                                
                          else{
                                for ( i = 0; i < n; i++ ){ // loop over captures
                                      if ( FROM(&movelist[i]) == bestfrom )
                                      if ( TO(&movelist[i])  == bestto ){
                                      SortVals[i] =  1000000;continue;
                                                                                                      }
                                      if ( is_promotion(&movelist[i] ) == 1 ) SortVals[i] = 4000;
                                      else
                                      SortVals[i] = 0;
                                      
                                      for ( int j = 2;j < movelist[i].l;j++ )
                                      SortVals[i] += MVALUE[(movelist[i].m[j]) >> 6];
                                                       } // loop over captures
                              }
      // IID

      int best_value = -INFINITY;
      int bestindex;
      int do_singular = 1; //
       if ( n == 1 && realdepth*2 < depth ){
       bestfrom = FROM( &movelist[0] );
       bestto = TO( &movelist[0] );
       goto NOIID;
                                                                    }
       if ( depth >= 4 ){
       if ( ((bestfrom == 0) && (bestto == 0)) || ( depth >= 6 && tr_depth < depth - 3 - 1 ) ){
       int tempalpha = alpha - 20*depth;
       tempalpha = MAX(-MATE,tempalpha);
       int tempbeta = beta + 20*depth;
       tempbeta = MIN(MATE,tempbeta);
       int new_depth;
       new_depth = depth - 3;
       bool ir = inSearch;
       inSearch = true;
       QuickSort( SortVals,movelist, 0,(n-1));
                     for( i = 0; i < n; i++){
                    domove(b,&movelist[i],color);
                    value = -PVSearch(b,new_depth,-tempbeta,-tempalpha,color^CC);
                    undomove(b,&movelist[i],color);
                    // restore HASH_KEY
                    HASH_KEY = L_HASH_KEY;
                    if (*play) return (0);
                    if ( value > best_value ){
                    	 best_value = value;
                    	 bestindex = i;
                    	 if ( value > tempalpha ){
                    	 	tempalpha = value;
                	  	 if ( value >= tempbeta )
               	  	 	break;
                                           	  	 	          }
                                                          }
                                                    } // for
                     inSearch = ir;
                 
                     if ( best_value < alpha && best_value <= -HASHMATE ) return best_value;
                     if ( best_value >= beta && best_value >= HASHMATE ) return best_value;
                     
                     bestfrom = FROM( &movelist[bestindex] );
                     bestto = TO( &movelist[bestindex] );
                     SortVals[bestindex] = 1000000;
                     if ( realdepth*2 >= depth && best_value < alpha - 175){
                     do_singular = 0; // best move is too bad to be singular
                                                                      }
                                                       }
                               }
 NOIID:
                int singular = 0;
                if ( do_singular == 1 && bestfrom != 0 && bestto != 0 ){
                if ( n == 1 )
                singular = 1;
                if ( n == 2 )
                singular = 1;
                         }
   // erase deeper killer moves
 
   memset(&killersf1[realdepth+2], 0, sizeof(int));
   memset(&killerst1[realdepth+2], 0, sizeof(int));
   memset(&killersf2[realdepth+2], 0, sizeof(int));
   memset(&killerst2[realdepth+2], 0, sizeof(int));
 
   int played_nb = 0;
   int played[MAXMOVES];
   best_value = -INFINITY; // preset maxvalue
   int newdepth;
   int ext;
   int prom; // promotion or not
   int bef_cap; //
   int aft_cap; //
   int extension; 
   int bestmovei;
 
   // Ok,let's look now at all moves and pick one with the biggest value
   while ( pick_next_move( &i,&SortVals[0], n ) != 0 ){
   prom = is_promotion( &movelist[i] );
   if ( (FROM( &movelist[i]) != bestfrom || TO( &movelist[i]) != bestto) )
   singular = 0;
   extension = 0;
    if ( capture || prom )
    bef_cap = (g_pieces[BLK_MAN] - g_pieces[WHT_MAN])*100 + (g_pieces[BLK_KNG] - g_pieces[WHT_KNG])*300;
   // domove
   domove(b,&movelist[i],color);
  
   if ( capture || prom ){ // try to extend
   aft_cap = (g_pieces[BLK_MAN] - g_pieces[WHT_MAN])*100 + (g_pieces[BLK_KNG] - g_pieces[WHT_KNG])*300;
   if (( bef_cap < 0 && aft_cap > 0 ) || ( bef_cap > 0 && aft_cap < 0 ) || ( aft_cap == g_root_mb ))
   if ( DO_EXTENSIONS )
   extension = 1;
                                       }
    newdepth = depth - 1 + MAX(extension,singular);
 if ( newdepth > 0 && ( FROM( &movelist[i] ) != bestfrom || TO( &movelist[i] ) != bestto ) ){
 	if ( newdepth <= 3 )
    value = -LowDepth(b,newdepth,-alpha ,color^CC);
    else
    value = -Search(b,newdepth,-alpha,color^CC,NodeCut, ( played_nb >= 5 ) ? true : false );
  	if ( value > alpha ){
    value = -PVSearch(b,newdepth,  -beta,-alpha,color^CC);
                                   }
                                       }
    else
    value = -PVSearch(b,newdepth,-beta,-alpha,color^CC);
    undomove(b,&movelist[i],color);
    played[played_nb] = i;
    played_nb++;
      // restore HASH_KEY
      HASH_KEY = L_HASH_KEY;
       if ( *play ) return (0);
      // update best value so far
      // and set alpha and beta bounds
       if (value > best_value){
       best_value = value;
 	  if ( value > alpha ){
       bestmovei = i;
       alpha = value;
       hashstore( value_to_tt(best_value), depth, FROM( &movelist[i] ) , TO( &movelist[i] ) , LOWER );
       if ( value >= beta ){
         if ( capture == 0 && is_promotion( &movelist[i] ) == 0 ){
         killer( FROM( &movelist[i] ) , TO( &movelist[i] ) ,realdepth,capture);
         hist_succ( FROM( &movelist[i] )   ,  TO( &movelist[i] ) ,depth,capture);
         for (i = 0; i < played_nb - 1; i++){
         int j = played[i];
         if ( is_promotion(&movelist[j] ) == 0 )
         history_bad( ((movelist[j].m[0]) & 0x3f),((movelist[j].m[1]) & 0x3f), depth);
                                                               }
                                                                                                         }
       return ( value );                                                                                                         
                                       }
                                }
                              }
                      }; // end main recursive loop of forallmoves
         if ( alpha != oldalpha ){
         bestfrom = ( movelist[bestmovei].m[0] ) & 0x3f;
         bestto = ( movelist[bestmovei].m[1] ) & 0x3f;
         if ( capture == 0 && is_promotion( &movelist[bestmovei] ) == 0 ){
         killer( bestfrom , bestto ,realdepth,capture);
         hist_succ( bestfrom , bestto  ,depth,capture);
                                                                                                                      }
         hashstore( value_to_tt(best_value), depth,bestfrom, bestto, EXACT );
         return best_value;
                                                }
         hashstore( value_to_tt(best_value), depth,0,0,UPPER);
         return best_value;
       }

          // Search is the search function for zero-width nodes
 static int Search(int b[46],int depth,int beta ,int color,int node_type, bool mcp ){
        // mate distance pruning
 
    if ( value_mated_in(realdepth) >= beta )
       return beta;

    if ( value_mate_in(realdepth + 1) < beta )
       return beta - 1;

                  /* time check */
                   
                  if ( !( ++nodes & 0x1fff ) )
                  if ( ((clock()-start) > maxtime*g_Panic) )
                	(*play) = 1;

                  /* return if calculation interrupt */
                  if (*play) return (0);

                
                  /* stop search if maximal search depth is reached */
                 if ( realdepth >= MAXDEPTH ) return evaluation(b,color,beta-1 ,beta);
                 
                 if ( realdepth > g_seldepth ) g_seldepth = realdepth; // new selective depth
                 	
                  /* check for database use */
                 
  unsigned int Pieces=g_pieces[BLK_MAN]+g_pieces[BLK_KNG]+g_pieces[WHT_MAN]+g_pieces[WHT_KNG];
 
  if ( (!EdNocaptures) && (Pieces <= EdPieces) )
   {
      int res = EdProbe(b,color);
      if (res != EdAccess::not_found)
      {
         if ( res != EdRoot[color] || !Reversible[realdepth - 1] )
         {
      if (res == EdAccess::win)  return ED_WIN - 100*Pieces;
      if (res == EdAccess::lose) return -ED_WIN + 100*Pieces;
      if (res == EdAccess::draw) return (0);
      MessageBox(0, "unknown value from EdAccess", "Error", 0);
         }
      }
   }
       U64 L_HASH_KEY = RepNum[realdepth] = HASH_KEY;  // local variable  for saving position's HASH_KEY
 
         // draw by triple repetition ?
         if ( g_pieces[WHT_KNG] && g_pieces[BLK_KNG] ){
   	    for (int i = 4; i <= realdepth; i += 2){
         if ( RepNum[ realdepth - i ] == HASH_KEY )
         return (0);
   		              }
                                                        }  
      
   int value;    
        	

   U8 bestfrom,bestto;
   U8 bestfrom2,bestto2;
   bestfrom =  bestto = 0; // best move's from-to squares
   bestfrom2 = bestto2 = 0;
   int try_mcp = 1;   // try forward pruning,initially enabled
   int tr_depth = 1024;
   // hashlookup
   if ( hashretrieve(&tr_depth,depth,&value,beta-1,beta,&bestfrom,&bestto,&bestfrom2,&bestto2, &try_mcp))
   return value;
   int i;
   struct move2 movelist[MAXMOVES];
   // check if the side to move has a capture
   int capture;
   capture = Test_Capture(b,color);   // is there a capture for the side to move ?
     
   if ( capture == 0 ){
      /* check for compressed ( without captures ) database use */
      if ( EdNocaptures && (Pieces <= EdPieces) ){
      int res = EdProbe(b,color);
      if (res != EdAccess::not_found)
      {
         if (res != EdRoot[color] || !Reversible[realdepth - 1])
         {
      if (res == EdAccess::win) return ED_WIN - 100*Pieces;
      if (res == EdAccess::lose) return -ED_WIN + 100*Pieces;
      if (res == EdAccess::draw) return (0);
      MessageBox(0, "unknown value from EdAccess", "Error", 0);
         }
      }
                                                                            }
                                 }
            
   int n;
   if ( capture ){
   n = Gen_Captures(b,movelist,color,capture);
                         }
   else{
   n = Gen_Moves(b,movelist,color);
   // if we have no moves and no captures :
   if ( n == 0 ){
   return (realdepth-MATE); // minus sign will be negated in negamax framework
                     };
         }
  /* enhanced transposition cutoffs: do every move and check if the resulting
 position is in the hashtable. if yes, check if the value there leads to a cutoff
 if yes, we don't have to search */
 
       if ( tr_depth < depth && depth >= ETCDEPTH && beta > -1500 ){
       U8 dummy;
       int dummy2;
       dummy = dummy2 = 0;
       int ETCvalue;
       for( i = 0; i < n; i++ ){
       // do move
       // domove(b,&movelist[i] );
       update_hash(&movelist[i]); // instead domove
       // do the ETC lookup:with reduced depth and changed color
       ETCvalue = -INFINITY;
       if (hashretrieve(&dummy2,depth-1,&ETCvalue,-beta,-beta+1 ,&dummy,&dummy,&dummy,&dummy,&dummy2)){
       // if one of the values we find is > beta we quit!
       if ( (-ETCvalue) >= beta ){
       // before we quit: restore all stuff
       // undomove(b,&movelist[i] );
       HASH_KEY = L_HASH_KEY;
       hashstore(value_to_tt(-ETCvalue),depth, FROM(&movelist[i]),TO( &movelist[i]),LOWER);
       return (-ETCvalue);
                                                }
                                                       } // if hashretrieve
      // undomove(b,&movelist[i] );
       HASH_KEY = L_HASH_KEY;
                                     } // for
                                               } // ETC
                         int SortVals[MAXMOVES];
                         // sort move list
                         // fill SortVals array with move's values
                    
                         int hindex;
                         if ( capture == 0 ){
                         for ( i = 0; i < n; i++ ){ // loop over moves
                         if ( FROM( &movelist[i]) == bestfrom ){
                         if ( TO( &movelist[i]) == bestto ){
                         SortVals[i] = 1000000;continue;
                                                                      }
                                                                    }
                         if ( FROM( &movelist[i]) == bestfrom2 ){
                         if ( TO( &movelist[i]) == bestto2 ){
                         SortVals[i] = 1000000 - 2;continue;
                                                                      }
                                                                    }
                                                                                                                                                          
                        if ( is_promotion(&movelist[i]) == 1 ){
                        SortVals[i] = 600000;continue;
                                                                                     }
                         if ( FROM( &movelist[i]) == killersf1[realdepth] ){
                         if ( TO( &movelist[i]) == killerst1[realdepth] ){
                         SortVals[i] = 600000 - 2;continue;
                                                                      }
                                                                    }
                                                                       
                          if ( FROM(&movelist[i]) == killersf2[realdepth] ){
	                     if ( TO(&movelist[i]) == killerst2[realdepth] ){
	                     SortVals[i] = 600000 - 4;continue;
                                                            }
                                                          }
                       
                        hindex = (SQUARE_TO_32(FROM(&movelist[i])) << 5) + SQUARE_TO_32(TO(&movelist[i]));
                        SortVals[i] = history_tabular[hindex];
                         
                          } // loop over moves
                                     
                                    }
                                
                          else{
                                      for ( i = 0; i < n; i++ ){ // loop over captures
                                      if ( FROM(&movelist[i]) == bestfrom )
                                      if ( TO(&movelist[i])  == bestto ){
                                      SortVals[i] =  1000000;continue;
                                                                                                      }
                                      if ( is_promotion(&movelist[i] ) == 1 ) SortVals[i] = 4000;
                                      else
                                      SortVals[i] = 0;
                                      
                                      for ( int j = 2; j < movelist[i].l;j++ )
                                      SortVals[i] += MVALUE[ (movelist[i].m[j]) >> 6];
                                                       } // loop over captures
                              }

        // IID
       
       int maxvalue;
       int bestindex;
       int do_singular = 1; //
       if ( n == 1 && realdepth*2 < depth ){
       bestfrom = FROM( &movelist[0] );
       bestto = TO( &movelist[0] );
       goto NOIID2;
                                                                    }
       if ( depth >= 5  ){
       if (( (bestfrom == 0) && (bestto == 0) ) || ( depth >= 6 && tr_depth < depth - 4 - 1)){
                    maxvalue = -INFINITY;
                    bool ir = inSearch;
                    inSearch = true;
                    int new_depth = depth - 4;
                    QuickSort( SortVals,movelist, 0,(n-1));
                    for( i = 0; i < n; i++){
                    domove(b,&movelist[i],color);
                    if ( new_depth <= 3 )
                    value = -LowDepth(b,new_depth,1 - beta , color^CC);
                    else
                    value = -Search(b,new_depth,1 - beta ,color^CC,NODE_OPP(node_type),false );
                    undomove(b,&movelist[i],color);
                    // restore HASH_KEY
                    HASH_KEY = L_HASH_KEY;
                    if (*play) return (0);
                    if ( value > maxvalue ){
                    	 maxvalue = value;
                    	 bestindex = i;
                    	 if ( value >= beta ) break;
                                                          }
                                                    } // for
                     inSearch = ir;
                     if ( maxvalue < beta && maxvalue <= -HASHMATE ) return maxvalue;
                     if ( maxvalue >= beta && maxvalue >= HASHMATE ) return maxvalue;
                  
                     bestfrom = FROM( &movelist[bestindex] );
                     bestto = TO( &movelist[bestindex] );
                     SortVals[bestindex] = 1000000;
                     if ( realdepth*2 >= depth && maxvalue < beta - 1 - 175 ){
                     do_singular = 0; // best move is too bad to be singular
                                                                        }
                                        }
                       }
  NOIID2:
                int singular = 0;
                if ( do_singular == 1 &&  bestfrom != 0 && bestto != 0 ){
                if ( n == 1 )
                singular = 1;
                if ( n == 2 )
                singular = 1;
                             }
       // ----------------------------------------------------------------------------------------------------
       // forward pruning
       // do not works in endgames
       // applied only at expected Cut nodes
       // -----------------------------------------------------------------------------------------------------
       
 int quick_eval;
 if ( beta > -1500 && beta < 1500 && mcp && ( realdepth > 4 ) && ( node_type == NodeCut ) && ( n > 1 ) && EARLY_GAME && try_mcp ){
 if ( !inSearch && (has_man_on_7th(b,color^CC) == 0 ) ){
         quick_eval = fast_eval( b, color );
         if ( quick_eval +2 >= beta ){
         int newdepth;
         int R = 3;
         newdepth = depth - R;
         if ( quick_eval - beta > 128 )
        	newdepth--;
         int played = 0;
         int max_moves;
         max_moves = ( g_pieces[(color|KING)] != 0 ) ? MAXMOVES : 6;
         U8 best_from,best_to;
              
               // loop 
               for ( i = 0 ; i < n; i++){
	                                Sort(i,n,SortVals,movelist);
	                                // do move
                                    domove(b,&movelist[i],color);
                                    if ( newdepth <= 0 )
                                    value = -QSearch(b,-beta - MCP_MARGIN ,-beta - MCP_MARGIN  + 1,color^CC);
                                    else{
                                    if ( newdepth <= 3 )
                                    value =  -LowDepth(b,newdepth,-beta - MCP_MARGIN  + 1 , color^CC );
                                    else
                                    value = -Search(b,newdepth, -beta - MCP_MARGIN  + 1 ,color^CC,played == 0 ? NodeAll : NodeCut, false);
                                           }
                                    //  undo move
                                    undomove(b,&movelist[i],color);
                                    // restore HASH_KEY
                                    HASH_KEY = L_HASH_KEY;
                                    if (*play) return (0);
                                    played++;
                                    if ( value >= beta + MCP_MARGIN ){
                                    best_from = FROM(&movelist[i]);
                                    best_to = TO(&movelist[i]);
                                    if ( capture == 0 && is_promotion( &movelist[i] ) == 0 ){
                                    hist_succ(best_from ,best_to ,depth,capture);
                                    killer(best_from ,best_to ,realdepth,capture);
                                                                                                    }
                                    return ( value );
                                                                                  }
                                    if ( played >= max_moves ) break;
                          } // for
                                       }
                                                 } // if
                                            } // if

   // erase deeper killer moves

   memset(&killersf1[realdepth+2], 0, sizeof(int));
   memset(&killerst1[realdepth+2], 0, sizeof(int));
   memset(&killersf2[realdepth+2], 0, sizeof(int));
   memset(&killerst2[realdepth+2], 0, sizeof(int));

   int played_nb = 0;
   int played[MAXMOVES];
  // maxvalue = -32767; // preset maxvalue
   int newdepth;
   int ext;
   int prom; // promotion or not
   int bef_cap; //
   int aft_cap; //
   int extension;
   int best_value;
   best_value = -INFINITY;
   U8 from,to;
   bool do_lmr_1,do_lmr_2;
   // Ok,let's look now at all moves and pick one with the biggest value
   while ( pick_next_move( &i,&SortVals[0], n ) != 0 ){
   	
     if ( (FROM(&movelist[i]) != bestfrom || TO(&movelist[i]) != bestto) )
   	singular = 0;
   	
   prom = is_promotion( &movelist[i] );
  
   extension = 0;

   if ( capture || prom )
   bef_cap =  (g_pieces[BLK_MAN] - g_pieces[WHT_MAN])*100 + (g_pieces[BLK_KNG] - g_pieces[WHT_KNG])*300;
   // domove
   domove(b,&movelist[i],color);
   if ( capture || prom ){ // try to extend
   aft_cap =  (g_pieces[BLK_MAN] - g_pieces[WHT_MAN])*100 + (g_pieces[BLK_KNG] - g_pieces[WHT_KNG])*300;
   if (( bef_cap < 0 && aft_cap > 0 ) || ( bef_cap > 0 && aft_cap < 0 ) || ( aft_cap == g_root_mb ))
   if ( DO_EXTENSIONS )
   if ( FROM( &movelist[i]) == bestfrom && TO( &movelist[i]) == bestto ) 
   extension = 1;
                                        }
   extension  = MAX ( extension , singular );
 	
   if ( played_nb > 1 && capture == 0 && prom == 0 && extension == 0  && !move_to_a1h8( &movelist[i] ) && !is_move_to_7( b,&movelist[i] ) ){
      from = FROM( &movelist[i] );
   	 to = TO( &movelist[i] );
      do_lmr_1 =  ( from != killersf1[realdepth] || to != killerst1[realdepth] ) ? true : false;
      do_lmr_2 =  ( from != killersf2[realdepth] || to != killerst2[realdepth] ) ? true : false;
   if ( do_lmr_1 && do_lmr_2 ){
   newdepth = depth - 3;
   if ( newdepth <= 0 ) 
   value = -QSearch(b, -beta  , 1 - beta ,color^CC);
   else{
   if ( newdepth <= 3 )
   value = -LowDepth( b,newdepth ,1 - beta,color^CC );
   else
   value = -Search(b,newdepth,1 - beta ,color^CC,NODE_OPP(node_type), false );
         }
   if ( value < beta ) goto DONE;
                                                      }
                                                    }
   newdepth = depth - 1 + extension;
   if ( newdepth <= 3 )
   value = -LowDepth( b,newdepth,1 - beta,color^CC );
   else
   value = -Search(b,newdepth,1 - beta ,color^CC,NODE_OPP(node_type), ( played_nb >= 5 ) ? true : false  );
    DONE:
    undomove(b,&movelist[i],color);
    played[played_nb] = i;
    played_nb++;
      // restore HASH_KEY
      HASH_KEY = L_HASH_KEY;
       if ( *play ) return (0);
      // update best value so far
      // and set alpha and beta bounds
       if (value >= beta){
         bestfrom = ( movelist[i].m[0] ) & 0x3f;
         bestto = ( movelist[i].m[1] ) & 0x3f;
       	if ( capture == 0 && is_promotion( &movelist[i] ) == 0 ){
         killer(bestfrom,bestto,realdepth,capture); // update killers
         hist_succ(bestfrom,bestto,depth,capture); // update history
         for (i = 0; i < played_nb - 1; i++){
         int j = played[i];
         if ( is_promotion(&movelist[j] ) == 0 )
         history_bad( ((movelist[j].m[0]) & 0x3f),((movelist[j].m[1]) & 0x3f),depth);
                                                               }
                                       }
         hashstore( value_to_tt(value), depth,bestfrom, bestto,LOWER);
         return value;
                                         }
         if ( value > best_value ) best_value = value;
              // if we were supposed to fail high but did not ...
         if ( node_type == NodeCut ) node_type = NodeAll;
            }; // end main recursive loop of forallmoves
     
      hashstore( value_to_tt( best_value ), depth,0, 0,UPPER);
      return best_value;
       }
  
static int LowDepth(int b[46],int depth,int beta ,int color ){
    // mate distance pruning
                    
    if ( value_mated_in(realdepth) >= beta )
       return beta;

    if ( value_mate_in(realdepth + 1) < beta )
       return beta - 1;
                  /* time check */
                   
                  if ( !( ++nodes & 0x1fff ) )
                  if ( ((clock()-start) > maxtime*g_Panic) )
                	(*play) = 1;

                  /* return if calculation interrupt */
                  if (*play) return (0);

                
                  /* stop search if maximal search depth is reached */
                 if ( realdepth >= MAXDEPTH ) return evaluation(b,color,beta-1 ,beta);
                 
                 if ( realdepth > g_seldepth ) g_seldepth = realdepth; // new selective depth
                 	
                  /* check for database use */
                 
  unsigned int Pieces=g_pieces[BLK_MAN]+g_pieces[BLK_KNG]+g_pieces[WHT_MAN]+g_pieces[WHT_KNG];
 
  if ( (!EdNocaptures) && (Pieces <= EdPieces) )
   {
      int res = EdProbe(b,color);
      if (res != EdAccess::not_found)
      {
         if ( res != EdRoot[color] || !Reversible[realdepth - 1] )
         {
      if (res == EdAccess::win)  return ED_WIN - 100*Pieces;
      if (res == EdAccess::lose) return -ED_WIN + 100*Pieces;
      if (res == EdAccess::draw) return (0);
      MessageBox(0, "unknown value from EdAccess", "Error", 0);
         }
      }
   }

         U64 L_HASH_KEY = RepNum[realdepth] = HASH_KEY;  // local variable  for saving position's HASH_KEY
        
         // draw by triple repetition ?
         if ( g_pieces[WHT_KNG] && g_pieces[BLK_KNG] ){
   	    for (int i = 4; i <= realdepth; i += 2){
         if ( RepNum[ realdepth - i ] == HASH_KEY )
         return (0);
   		              }
                                                        }  

   int value;

   U8 bestfrom,bestto;
   U8 bestfrom2,bestto2;
   bestfrom =  bestto = 0; // best move's from-to squares
   bestfrom2 =  bestto2 = 0;
   int tr_depth = -1024;
   int try_mcp = 1;
   // hashlookup
   if ( hashretrieve(&tr_depth,depth,&value, beta - 1,beta,&bestfrom,&bestto,&bestfrom2,&bestto2,&try_mcp))
   return value;

   // razoring
   if (  bestfrom == 0 && bestto == 0 && realdepth > 8 && NOT_ENDGAME &&  !inSearch && abs(beta) < 1500  ){
   int MARGIN[4] = { 2048, 116 , 150, 270 };
   if ( fast_eval( b, color ) < beta - MARGIN[depth] ){
   nodes--;
   value =  QSearch(b, beta - 1 - MARGIN[depth]   , beta - MARGIN[depth]  ,color);
   if ( *play ) return (0);
   if ( value < beta - MARGIN[depth] ){
   hashstore( value, depth, 0 , 0, UPPER );
   return value;
                                                               }
                     }
       };
 
   int i;
   struct move2 movelist[MAXMOVES];
   // check if the side to move has a capture
   int capture;
   capture = Test_Capture(b,color);   // is there a capture for the side to move ?
   if ( capture == 0 ){
      /* check for compressed ( without captures ) database use */
      if ( EdNocaptures && (Pieces <= EdPieces) ){
      int res = EdProbe(b,color);
      if (res != EdAccess::not_found)
      {
         if (res != EdRoot[color] || !Reversible[realdepth - 1])
         {
      if (res == EdAccess::win) return ED_WIN - 100*Pieces;
      if (res == EdAccess::lose) return -ED_WIN + 100*Pieces;
      if (res == EdAccess::draw) return (0);
      MessageBox(0, "unknown value from EdAccess", "Error", 0);
         }
      }
                                                                            }
                                 }
   int n;
   if ( capture ){
   n = Gen_Captures(b,movelist,color,capture);
                        }
   else{
   n = Gen_Moves(b,movelist,color);
   // if we have no moves and no captures :
   if ( n == 0 ){
   return (realdepth-MATE); // minus sign will be negated in negamax framework
                     };
         }
                         int SortVals[MAXMOVES];
                         // sort move list
                         // fill SortVals array with move's values
                    
                         int hindex;
                         if ( capture == 0 ){
                         for ( i = 0; i < n; i++ ){ // loop over moves
                         if ( FROM( &movelist[i]) == bestfrom ){
                         if ( TO( &movelist[i]) == bestto ){
                         SortVals[i] = 1000000;continue;
                                                                      }
                                                                    }
                         if ( FROM( &movelist[i]) == bestfrom2 ){
                         if ( TO( &movelist[i]) == bestto2 ){
                         SortVals[i] = 1000000 - 2;continue;
                                                                      }
                                                                    }
                        if ( is_promotion(&movelist[i]) == 1 ){
                        SortVals[i] = 600000;continue;
                                                                                     }
                         if ( FROM( &movelist[i]) == killersf1[realdepth] ){
                         if ( TO( &movelist[i]) == killerst1[realdepth] ){
                         SortVals[i] = 600000 - 2;continue;
                                                                      }
                                                                    }
                                                                       
                         if ( FROM( &movelist[i]) == killersf2[realdepth] ){
	                     if ( TO( &movelist[i]) == killerst2[realdepth] ){
	                     SortVals[i] = 600000 - 4;continue;
                                                            }
                                                          }
                    
                        hindex = (SQUARE_TO_32( FROM( &movelist[i])) << 5) + SQUARE_TO_32(TO( &movelist[i]));
                         SortVals[i] = history_tabular[hindex];
                         
                          } // loop over moves
                                     
                                    }
                                
                          else{
                                      for ( i = 0; i < n; i++ ){ // loop over captures
                                      if ( FROM( &movelist[i]) == bestfrom )
                                      if ( TO( &movelist[i])  == bestto ){
                                      SortVals[i] =  1000000;continue;
                                                                                                      }
                                      if ( is_promotion(&movelist[i] ) == 1 ) SortVals[i] = 4000;
                                      else
                                      SortVals[i] = 0;
                                      
                                      for ( int j = 2;j < movelist[i].l;j++ )
                                      SortVals[i] += MVALUE[ (movelist[i].m[j]) >> 6 ];
                                                       } // loop over captures
                              }

   // erase deeper killer moves
   
   memset(&killersf1[realdepth+2], 0, sizeof(int));
   memset(&killerst1[realdepth+2], 0, sizeof(int));
   memset(&killersf2[realdepth+2], 0, sizeof(int));
   memset(&killerst2[realdepth+2], 0, sizeof(int));
 
   int played_nb = 0;
   int played[MAXMOVES];
   int best_value;
   best_value = -INFINITY;
   int new_depth;
   // Ok,let's look now at all moves and pick one with the biggest value
   while ( pick_next_move( &i,&SortVals[0], n ) != 0 ){
      new_depth = depth - 1;
      // domove
      domove(b,&movelist[i],color);

      if ( new_depth <= 0 )
      value = -QSearch(b,-beta,1 - beta,color^CC);
      else
      value = -LowDepth(b,new_depth,1 - beta ,color^CC );
      undomove(b,&movelist[i],color);
      played[played_nb] = i;
      played_nb++;
      // restore HASH_KEY
      HASH_KEY = L_HASH_KEY;
       if ( *play ) return (0);
      // update best value so far
      // and set alpha and beta bounds
       if (value >= beta){
         bestfrom = ( movelist[i].m[0] ) & 0x3f;
         bestto = ( movelist[i].m[1] ) & 0x3f;
       	if ( capture == 0 && is_promotion( &movelist[i] ) == 0 ){
         killer(bestfrom,bestto,realdepth,capture); // update killers
         hist_succ(bestfrom,bestto,depth,capture); // update history
         for (i = 0; i < played_nb - 1; i++){
         int j = played[i];
         if ( is_promotion(&movelist[j] ) == 0 )
         history_bad( ((movelist[j].m[0]) & 0x3f),((movelist[j].m[1]) & 0x3f),depth);
                                                               }
                                       }
         hashstore( value_to_tt(value), depth,bestfrom, bestto,LOWER);
         return value;
                                         }
         if ( value >= best_value )
         best_value = value;
               }; // end main recursive loop of forallmoves
      hashstore( value_to_tt( best_value ), depth,0, 0,UPPER);
      return best_value;
      }

static void hashstore(int value, int depth,U8 best_from, U8 best_to,int f){
     //
     //
                  int age;
                  int oldest = 0;
                  TEntry *entry, *replace;
                  U32 lock;
                  register unsigned int i;
              
                   entry = replace = ttable +(U32) (HASH_KEY & MASK); // first entry
                   lock = (HASH_KEY >> 32); // use the high 32 bits as lock
                
                   for( i = 0;i < REHASH;i++,entry++){
                   if ( ((entry->m_lock) == 0) || ((entry->m_lock) == lock) ){ // empty or hash hit => update existing entry
                   	                 // don't overwrite entry's best move with 0
                                       if ( ( best_from == 0 ) && ( best_to == 0 ) ){
                                       best_from = (entry->m_best_from);
                                       best_to = (entry->m_best_to);
                                                                                                             }
                                       entry->m_valuetype = f;
                                       
                                      if (( entry->m_best_from != best_from ) || ( entry->m_best_to != best_to )){
                                       entry->m_best_from2 = entry->m_best_from;
                                       entry->m_best_to2 = entry->m_best_to;
                                       entry->m_best_from = best_from;
                                       entry->m_best_to = best_to;
                                              }                                       
                                       	
                                       entry->m_lock = lock;
                                       entry->m_depth = depth;
                                       entry->m_value = value;
                                       entry->m_age = searches_performed_in_game;
                                       return;
                                                } // empty or hash hit
  age = ((( searches_performed_in_game - entry->m_age ) & 255 ) * MAXDEPTH   + MAXDEPTH - (entry->m_depth ));
                        if (age > oldest){
                        oldest = age;
                        replace = entry;
                                                  }
                            } // rehash loop
                                 
                                       replace->m_valuetype = f;
                                       entry->m_best_from = best_from;
                                       entry->m_best_to = best_to;
                                       entry->m_best_from2 = 0;
                                       entry->m_best_to2 = 0;
                                       replace->m_lock = lock;
                                       replace->m_depth = depth;
                                       replace->m_value = value;
                                       replace->m_age = searches_performed_in_game;
                             }


static int hashretrieve(int *tr_depth,int depth,int *value,int alpha,int beta,U8 *best_from,U8 *best_to,U8 *best_from2,U8 *best_to2, int *try_mcp){
   //
   //
                  TEntry *beste,*entry;
                  register unsigned int i;
                  U32 lock;
                  int found = 0;
            
                  entry = ttable + (U32)(HASH_KEY & MASK); // first entry
                  lock = (HASH_KEY >> 32); // use the high 32 bits as lock
             
                  for( i = 0;( (i < REHASH ) && (found == 0) ) ; i++,entry++ ){
                  if ( (entry->m_lock) == lock){
                  found++;
                  beste = entry;
                                                                 }
                     } // for

           if ( found == 0 ){
           return (0);
                                      }
       
           int v = (beste->m_value);
           if (abs(v) >= HASHMATE){
           if (v > 0) v -= realdepth;
           if (v < 0) v += realdepth;
                                                        }
      // if we are searching with a higher remaining depth than what is in the hashtable then
      // all we can do is set the best move for move ordering
      if ( ( beste->m_depth ) >= depth ){
          
     // we have sufficient depth in the hashtable to possibly cause a cutoff.
     // if we have a lower bound, we might get a cutoff
   
     if ( ((beste->m_valuetype) & LOWER) != 0){
     // the value stored in the hashtable is a lower bound, so it's useful
                  if ( v >= beta ){
                  // value > beta: we can cutoff!
                  *value = v;
                  beste->m_age = searches_performed_in_game; // to avoid aging
                  return (1);
                                          }
                                                                             };
 
      // if we have an upper bound, we can get a cutoff
      if ( ((beste->m_valuetype) & UPPER) != 0){
      // the value stored in the hashtable is an upper bound, so it's useful
                  if( v <= alpha ){
                  // value < alpha: we can cutoff!
                  *value = v;
                  beste->m_age = searches_performed_in_game; // to avoid aging
                  return (1);
                                           }
                                                                    };
                                                      }
         // ***********************************************************************************
         // use mate values
         
         if ( v >= MAX(HASHMATE, beta)){
         	
         if ( ((beste->m_valuetype) & LOWER) != 0){
                  if ( v >= beta ){
                  *value = v;
                   beste->m_age = searches_performed_in_game;
                   return (1);
                                          }
                                                                                }
                                                                    }
                                                                    
        if ( v <= MIN(-HASHMATE, alpha)){
                
        if ( ((beste->m_valuetype) & UPPER) != 0){
                  if( v <= alpha ){
                  *value = v;
                   beste->m_age = searches_performed_in_game;
                   return (1);
                                            }
                                                                    }
                                                                }
           // forward pruning switch
           if ( ( (beste->m_valuetype) & UPPER) != 0 )
           if ( (beste->m_depth) >  ( depth >= 15 ) ? ( depth - 4 - 2 ) : ( depth - 3 - 2 ) )
           if ( v <= beta )
           *try_mcp = 0; // don't try forward pruning
               // set best move
          *best_from = (beste->m_best_from);
          *best_to = (beste->m_best_to);
          *best_from2 = (beste->m_best_from2);
          *best_to2 = (beste->m_best_to2);
          *tr_depth = (beste->m_depth);
      return (0);
                     }

static int value_to_tt( int value ){
      // modify value
      if ( value < -HASHMATE ){
      value -= realdepth;
      if ( value < -MATE ) value = -MATE;
      return ( value );
                                                   }
      if ( value > HASHMATE ){
  	 value += realdepth;
      if ( value > MATE ) value = MATE;
      return value;
                                                  }
      return ( value );                                                  
          }

static void retrievepv( int b[46], char *pv, int color){
   // gets the pv from the hashtable
   // get a pv string:
   int n;
   int i;
   U8 bestfrom;
   U8 bestto;
   U8 bestfrom2;
   U8 bestto2;
   int bestindex = 0;
   struct move2 movelist[MAXMOVES];
   int dummy = 0;
   char pvmove[256];
   int count = 0;
   int copy[46];
   // original board b[46] needs not to be changed
   for ( i=0;i<46;i++ )
   copy[i] = b[i];
   bestfrom = 0;
   bestto = 0;
   bestfrom2 = 0;
   bestto2 = 0;
   sprintf(pv,"");
   init_piece_lists(copy);
   HASH_KEY = Position_to_Hashnumber(copy,color);
   hashretrieve(&dummy,100, &dummy, dummy, dummy, &bestfrom, &bestto,&bestfrom2,&bestto2, &dummy);
   while( (bestfrom != 0) && (bestto != 0) && (count<10)){
      n = Gen_Captures( copy, movelist, color,1);
      if(!n)
      n = Gen_Moves( copy, movelist, color);
      if (!n) return;
      for ( i = 0; i < n ; i++ ){
      if ( (U8)(( movelist[i].m[0] ) & 0x3f) == bestfrom &&  (U8)(( movelist[i].m[1] ) & 0x3f) == bestto ){
      bestindex = i;
      break;
                                           }
               }
      
      movetonotation(movelist[bestindex],pvmove);
      domove2( copy, &movelist[bestindex],color );
      strcat(pv," ");
      strcat(pv,pvmove);
      color = color^CC;
      HASH_KEY = Position_to_Hashnumber(copy,color);
      // look up next move
      bestfrom = 0;
      bestto = 0;
      bestfrom2 = 0;
      bestto2 = 0;
      hashretrieve(&dummy,100, &dummy, dummy, dummy, &bestfrom, &bestto, &bestfrom2,&bestto2, &dummy);
      count++;
      }
             }


static void __inline killer( U8 bestfrom,U8 bestto,int realdepth,int capture){
//
     if ( capture ) return;
     if ( ( bestfrom == killersf1[realdepth] ) && ( bestto == killerst1[realdepth] ) )
 	return;
       killersf2[realdepth] = killersf1[realdepth];
       killerst2[realdepth] = killerst1[realdepth];
       killersf1[realdepth] = bestfrom;
       killerst1[realdepth] = bestto;
    return;
}


static void hist_succ(U8 from,U8 to,int depth,int capture ){
//
  if ( capture ) return;
  int hindex = (SQUARE_TO_32(from) <<5) + SQUARE_TO_32(to);
  int sv =  history_tabular[hindex] ;
  history_tabular[hindex] = sv + depth*depth;
  if ( history_tabular[hindex] > MAXHIST )
  	for ( int p = 0; p <= 1023 ; p++ )
   	history_tabular[p] = ( history_tabular[p] ) / 2;
       }

static void history_bad( U8 from,U8 to,int depth ){
  int hindex = (SQUARE_TO_32(from) <<5) + SQUARE_TO_32(to);
  int sv =  history_tabular[hindex] ;
  history_tabular[hindex] = sv - depth*depth;
  if ( history_tabular[hindex] < -MAXHIST )
  	for ( int p = 0; p <= 1023 ; p++ )
   	history_tabular[p] = ( history_tabular[p] ) / 2;
}

static void ClearHistory(void){
     // clear previous History before each new game
     for ( int i = 0; i <= 1023;i++){
     history_tabular[i] = 0;
                                              }
                                               
     // also clear the killer moves
     for ( int i = 0;i <= (MAXDEPTH + 1);i++ ){
     killersf1[i] = 0;
     killerst1[i] = 0;
     killersf2[i] = 0;
     killerst2[i] = 0;
                                                                }
     // clear repetition checker
     memset(RepNum,0,MAXDEPTH*sizeof(U64));
      }
 
static int has_man_on_7th(int b[46],int color){
  // color has passer
  if ( color == WHITE ){
     if (( b[13] == WHT_MAN ) && ( b[8] == FREE )) return (1);
     if (( b[12] == WHT_MAN ) && (( b[7] == FREE ) || ( b[8] == FREE ))) return (1);
    	if (( b[11] == WHT_MAN ) && (( b[6] == FREE ) || ( b[7] == FREE ))) return (1);
	if (( b[10] == WHT_MAN ) && (( b[5] == FREE ) || ( b[6] == FREE ))) return (1);
	return (0);	
                                      }
  else{
     if (( b[32] == BLK_MAN ) && ( b[37] == FREE )) return (1);
     if (( b[33] == BLK_MAN ) && (( b[37] == FREE ) || ( b[38] == FREE ))) return (1);
    	if (( b[34] == BLK_MAN ) && (( b[38] == FREE ) || ( b[39] == FREE ))) return (1);
	if (( b[35] == BLK_MAN ) && (( b[39] == FREE ) || ( b[40] == FREE ))) return (1);
	return (0);
       }
  }

static int fast_eval(int b[46], int color){
// returns material balance at given node + PST

    int nbm = g_pieces[BLK_MAN];
    int nbk = g_pieces[BLK_KNG];
    int nwm = g_pieces[WHT_MAN];
    int nwk = g_pieces[WHT_KNG];

    if ( (nbm == 0) && (nbk == 0) )  return ( realdepth - MATE);
    if ( (nwm == 0) && (nwk == 0) ) return ( realdepth - MATE);
           int v1 = 100 * nbm + 300 * nbk;
           int v2 = 100 * nwm + 300 * nwk;  
           int  eval = v1 - v2; // total evaluation
     
           int White = nwm + nwk; // total number of white pieces
           int Black = nbm + nbk;     // total number of black pieces
     
           // draw situations
           if ( nbm == 0 && nwm == 0 && abs( nbk - nwk) <= 1 ){ 
           return (0); // only kings left
                      }
           if ( ( eval > 0 ) && ( nwk > 0 ) && (Black < (nwk+2)) ){
           return (0); // black cannot win
                                          }

           if ( ( eval < 0 ) && (nbk > 0) && (White < (nbk+2)) ){
           return (0); //  white cannot win
                                           }
        
       
          int opening = 0;
          int endgame = 0;
          int square;
          unsigned int i;
            //a piece of code to encourage exchanges
		  //in case of material advantage:
		      // king's balance
            if ( nbk != nwk){
            if ( nwk == 0 ){
            	    if ( nwm <= 4 ){
                 endgame += 50;
                 if ( nwm <= 3 ){
                 endgame += 100;
                 if ( nwm <= 2 ){
                 endgame += 100;
                 if ( nwm <= 1 )
                 endgame += 100;	
                                          }
                                        }
                                       }
                                      }
             if ( nbk == 0 ){
            	 if ( nbm <= 4 ){
                 endgame -= 50;
                 if ( nbm <= 3 ){
                 endgame -= 100;
                 if ( nbm <= 2 ){
                 endgame -= 100;
                 if ( nbm <= 1 )
                 endgame -= 100;	
                                          }
                                        }
                                      }
                                  }  
                         } 
           else{           
           if ( (nbk == 0) && (nwk == 0) )
           eval += 250*( v1 - v2 ) / ( v1 + v2 ); 
           if ( nbk + nwk != 0 )
           eval += 100*( v1 - v2 ) / ( v1 + v2 );
                 }
         
     static U8 PST_man_op[41] = {0,0,0,0,0,   // 0 .. 4
                             15,40,42,45,0,              // 5 .. 8 (9)
                             12,38,36,15,                     // 10 .. 13
                             28,26,30,20,0,               // 14 .. 17 (18)
                             18,26,36,28,                    // 19 .. 22
                             32,38,10,18,0,                // 23 .. 26 (27)
                             18,22,24,20,                 //  28 .. 31
                             26,0,0,0,0,                      // 32 .. 35 (36)
                             0,0,0,0                          // 37 .. 40
                                       };
                                       
 static U8 PST_man_en[41] = {0,0,0,0,0,  // 0 .. 4
                             0,2,2,2,    0,                  // 5 .. 8 (9)
                             4,4,4,4,                     // 10 .. 13
                             6,6,6,6,    0,               // 14 .. 17 (18)
                             10,10,10,10,                  // 19 .. 22
                             16,16,16,16,   0,              // 23 .. 26 (27)
                             22,22,22,22,                //  28 .. 31
                             30,0,0,0,         0,            // 32 .. 35 (36)
                             0,0,0,0                        // 37 .. 40
                                       };     
 
         for ( i = 1;i <= num_bpieces;i++){
    	      if ( ( square = p_list[BLACK][i] )  == 0 ) continue;
          if ( ( b[square] & MAN ) != 0 ){     // black man
          opening += PST_man_op[square];
          endgame += PST_man_en[square];
                                                            }
                                                                  }
                                                                 
           for ( i = 1;i <= num_wpieces;i++){
           if ( (square = p_list[WHITE][i]) ==0 ) continue;
           if ( ( b[square] & MAN ) != 0 ){    // white man
           opening -= PST_man_op[45 - square];
           endgame -= PST_man_en[45 - square];
                                                             }
                                                               }
         
           // phase mix
           // smooth transition between game phases
           int phase = nbm + nwm - nbk - nwk;
           if ( phase < 0 ) phase = 0;
           int antiphase = 24 - phase;
           eval += ((opening * phase + endgame * antiphase )/24);
           if ( ( White + Black < 8 ) && nbk != 0 && nwk != 0 && nbm != 0 && nwm != 0 ){
    	       if ( abs(nbm - nwm ) <= 1  && abs(nbk - nwk ) <= 1 && abs( White - Black ) <= 1 ){
    	         	eval /= 2;
             }
    		  }
           // negamax formulation requires this:
           if ( color == BLACK ){
           return (eval);
                                               }
           else{
           return (-eval);
                 }
   }

static int __inline is_promotion(struct move2 *move){
//
//
  if ( ((move->m[0]) >> 6) == ((move->m[1]) >> 6) )
  return (0);
  return (1);
}

static int is_blk_kng( int b[46] ){
int retval = 0;
int i;
   for ( i = 5; i < 41 ; i += 5 ){
    if ( b[i] == WHT_KNG ) return(0);
    if ( b[i] == WHT_MAN ) return(0);	
    if ( b[i] == BLK_MAN ) return(0);
    if ( b[i] == BLK_KNG ) retval = 1;
                                          }
   return retval;
                                                    }

                                                    
static int is_blk_kng_1( int b[46] ){
// 8-32
int i;
int cnt = 0;
   for ( i = 8; i <= 32 ; i += 4 )
   if ( b[i] == BLK_KNG ) cnt++;
   
   return cnt;
                                                         }
                                                         
static int is_wht_kng_1( int b[46] ){
// 8-32
int i;
int cnt = 0;
   for ( i = 8; i <= 32 ; i += 4 )
   if ( b[i] == WHT_KNG ) cnt++;
   
   return cnt;
                                                          }
                                                          
static int is_blk_kng_2( int b[46] ){
// 13-37
int i;
int cnt = 0;
   for ( i = 13; i <= 37 ; i += 4 )
   if ( b[i] == BLK_KNG ) cnt++;
   
    return cnt;
                                                         }
                                                         
static int is_wht_kng_2( int b[46] ){
// 13-37
int i;
int cnt = 0;
   for ( i = 13; i <= 37 ; i += 4 )
   if ( b[i] == WHT_KNG ) cnt++;
   
    return cnt;
    
                                                          }
                                                          
static int is_blk_kng_3( int b[46] ){
// 14-39
int i;
int cnt = 0;
   for ( i = 14; i <= 39 ; i += 5 )
   if ( b[i] == BLK_KNG ) cnt++;
   
   return cnt;
                                                         }  
 static int is_wht_kng_3( int b[46] ){
// 14-39
int i;
int cnt = 0;
   for ( i = 14; i <= 39 ; i += 5 )
   if ( b[i] == WHT_KNG ) cnt++;
   
  return cnt;
                                                         }  
                                                         
static int is_blk_kng_4( int b[46] ){
// 6-31
int i;
int cnt = 0;
   for ( i = 6; i <= 31 ; i += 5 )
   if ( b[i] == BLK_KNG ) cnt++;
   
   return cnt;
                                                         }  
                                                         
static int is_wht_kng_4( int b[46] ){
// 6-31
int i;
int cnt = 0;
   for ( i = 6; i <= 31 ; i += 5 )
   if ( b[i] == WHT_KNG ) cnt++;
   	
   return cnt;
                                                         }  
                                                  	                                          
static int is_wht_kng( int b[46] ){
int retval = 0;
int i;
   for ( i = 5; i < 41 ; i += 5 ){
    if ( b[i] == WHT_MAN ) return(0);	
    if ( b[i] == BLK_MAN ) return(0);
    if ( b[i] == BLK_KNG ) return(0);
    if ( b[i] == WHT_KNG ) retval = 1;
                                              }
   return retval;
                                   }


static int pick_next_move( int *marker,int SortVals[MAXMOVES],int n ){
     // a function to give pick the top move order, one at a time on each call.
     // Will return 1 while there are still moves left, 0 after all moves
     // have been used

          int best = -1000000;
          int i;
          *marker = -32767;

          for ( i = 0; i < n; i++ ){
                if ( SortVals[i] > best ){
                     *marker = i;
                      best = SortVals[i];
                                                    }
                                             }

        if ( *marker > -32767 ){
             SortVals[*marker] = -1000000;
             return (1);
                                             }
         return (0);
     }


static void Sort( int start,int num,int SortVals[MAXMOVES],struct move2 movelist[MAXMOVES] ){
        // do a linear search through the current ply's movelist starting at start
        // and swap the best one with start

        int i,j;
        int Max;
        Max = SortVals[start];
        j = start;
        
        for( i = (start+1); i < num; i++){
          if( SortVals[i]  > Max){
              Max = SortVals[i];
              j  = i; // best index
                                              }
                                                    }

        if ( Max != SortVals[start] ){ // swap
        struct move2 m = movelist[start];
        movelist[start] = movelist[j];
        movelist[j] = m;
        int tmp = SortVals[start];
        SortVals[start] = SortVals[j];
        SortVals[j] = tmp;
                                                    }
          }

static void QuickSort( int SortVals[MAXMOVES],struct move2 movelist[MAXMOVES], int inf, int sup){
    // quick sort algorithm used to sort movelist
        int pivot;
        register int i,j;
        int swap;
        struct move2 temp;
        i = inf;
        j = sup;
        pivot = SortVals[(i+j)/2];
   do {
      while (SortVals[i] > pivot) i++;
      while (SortVals[j] < pivot) j--;
      if (i<j) {
         swap = SortVals[i];
         SortVals[i] = SortVals[j];
         SortVals[j] = swap;
         temp = movelist[i];
         movelist[i] = movelist[j];
         movelist[j] = temp;
      }
      if (i<=j) {
         i++;
         j--;
      }
   } while (i<=j);
    if (inf<j) QuickSort(SortVals,movelist,inf,j); // recurse
    if (i<sup) QuickSort(SortVals,movelist,i,sup); // recurse
}

static bool is_move_to_7( int b[46],struct move2 *move )
{
// unstopable move to 7th rank is considered as dangerous
unsigned int piece,to;
piece = (move->m[0]) >> 6;    // moving piece
to = (move->m[1]) & 0x3f;     // destination square

if ( piece == BLK_MAN ){
if ( to > 31 && to < 36 ){
	if ( to == 32 ){
	if ( b[37] == FREE )
	return true;
	return false;
	                      }
	if ( to == 33 ){
	if (( b[37] == WHT_MAN ) && ( b[29] == FREE )){
     if ( b[32] == BLK_MAN ) return true;
	return false;
	                  }
	if (( b[38] == WHT_MAN ) && ( b[28] == FREE ))
	return false;
	if (( b[28] == WHT_MAN ) && ( b[38] == FREE ))
	return false;
	if (( b[29] == WHT_MAN ) && ( b[37] == FREE ))
	return false;
	return true;
	                      }
	if ( to == 34 ){
	if (( b[38] == WHT_MAN ) && ( b[30] == FREE ))
	return false;
	if (( b[39] == WHT_MAN ) && ( b[29] == FREE ))
	return false;
	if (( b[29] == WHT_MAN ) && ( b[39] == FREE ))
	return false;
	if (( b[30] == WHT_MAN ) && ( b[38] == FREE ))
	return false;
	return true;
	                      }
	if ( to == 35 ){
	if (( b[40] == WHT_MAN ) && ( b[30] == FREE ))
	return false;
	if (( b[39] == WHT_MAN ) && ( b[31] == FREE ))
	return false;
	if (( b[30] == WHT_MAN ) && ( b[40] == FREE ))
	return false;
	if (( b[31] == WHT_MAN ) && ( b[39] == FREE ))
	return false;
	return true;
	                      }
	                    }
             }
	                  
if ( piece == WHT_MAN ){
if ( to > 9 && to < 14 ){
    if ( to == 10 ){
    	if (( b[5] == BLK_MAN ) && ( b[15] == FREE ))
    	return false;
	if (( b[6] == BLK_MAN ) && ( b[14] == FREE ))
    	return false;
 	if (( b[14] == BLK_MAN ) && ( b[6] == FREE ))
    	return false;
	if (( b[15] == BLK_MAN ) && ( b[5] == FREE ))
    	return false;
    	return true;
                          }
    if ( to == 11 ){
    	if (( b[6] == BLK_MAN ) && ( b[16] == FREE ))
    	return false;
    	if (( b[7] == BLK_MAN ) && ( b[15] == FREE ))
    	return false;
    	if (( b[15] == BLK_MAN ) && ( b[7] == FREE ))
    	return false;
	if (( b[16] == BLK_MAN ) && ( b[6] == FREE ))
    	return false;
    	return true;
                           }                    
     if ( to == 12 ){
    	if (( b[7] == BLK_MAN ) && ( b[17] == FREE ))
    	return false;
    	if (( b[8] == BLK_MAN ) && ( b[16] == FREE )){
    	if ( b[13] == WHT_MAN ) return true;
    	return false;
                                                                                    }
    	if (( b[16] == BLK_MAN ) && ( b[8] == FREE ))
    	return false;
	if (( b[17] == BLK_MAN ) && ( b[7] == FREE ))
    	return false;
    	return true;
                           }                    
    if ( to == 13 ){
    	if ( b[8] == FREE )
    	return true;
    	return false;
                          }                    
            }
      }
   return (false);
}

static bool move_to_a1h8( struct move2 *move ){
//
 unsigned int from,piece,to;
 piece = (move->m[0]) >> 6; // moving piece
 if (( piece & KING ) != 0){ // king move
 to = (move->m[1]) & 0x3f;  // destination square
 from = (move->m[0]) & 0x3f; // from square
 if (( to % 5 == 0 ) && ( from % 5 != 0 ))
    return (true);
                                          }
    return (false);
}


static void  init_piece_lists( int b[46] ){
  //
int color,i,j;

num_wpieces = 0;
num_bpieces = 0;

g_pieces[BLK_MAN] = 0;
g_pieces[BLK_KNG] = 0;
g_pieces[WHT_MAN] = 0;
g_pieces[WHT_KNG] = 0;

  for ( i=0;i<=40;i++ )
       indices[i] = 0;

  for ( i=0;i<=15;i++ ){
       p_list[WHITE][i] = 0;
       p_list[BLACK][i] = 0;
                                   }
                       
  for ( i=5;i<=40;i++ ){
       if  ( ( b[i] != OCCUPIED ) && ( b[i] != FREE ) ){
       g_pieces[b[i]]++;
       color = ( b[i] & WHITE ) ? WHITE:BLACK;
       if ( color == WHITE ){
       num_wpieces += 1;
       p_list[WHITE][num_wpieces] = i;
       indices[i] = num_wpieces;
                                          }
       else{
       num_bpieces += 1;
       p_list[BLACK][num_bpieces] = i;
       indices[i] = num_bpieces;
             }
                                                                                    }
       else
       indices[i] = 0;
                                 }
g_root_mb = (g_pieces[BLK_MAN] - g_pieces[WHT_MAN])*100 + (g_pieces[BLK_KNG] - g_pieces[WHT_KNG])*300;
}


static void Perft(int *b,int color,unsigned depth){
   // perfomance test
   register unsigned int capture,n;
   int i;
   capture = Test_Capture(b,color);
   if ( capture ){
     struct move2 movelist[MAXCAPTURES];
     n = Gen_Captures(b,movelist,color,capture);
     --depth;
     for ( i = n-1; i >=0 ;i--){
          domove2(b,&movelist[i],color );
          if ( depth ) Perft(b,color^CC,depth);
          else  ++PerftNodes;
          undomove(b,&movelist[i],color );
                              }
                         }
   else{
     struct move2 movelist[MAXMOVES];
     n = Gen_Moves(b,movelist,color);
     --depth;
     for ( i = n-1;i >= 0 ;i--){
          domove2(b,&movelist[i],color );
          if ( depth ) Perft(b,color^CC,depth);
          else  ++PerftNodes;
          undomove(b,&movelist[i],color );
                         }
          }
  }


static int EdProbe(int c[46],int color)
{
   if (!ED) return EdAccess::not_found;

   unsigned i;
   EdAccess::EdBoard1 b;
   static const int Map_32_to_45[32] = {
       8,  7,  6,  5,
      13, 12, 11, 10,
      17, 16, 15, 14,
      22, 21, 20, 19,
      26, 25, 24, 23,
      31, 30, 29, 28,
      35, 34, 33, 32,
      40, 39, 38, 37
   };

   if (color == WHITE)
   {
      for (i = 0; i < 32; i++)
      {
         switch (c[Map_32_to_45[i]])
         {
            case FREE           : b.board[i] = EdAccess::empty; break;
            case WHITE | MAN : b.board[i] = EdAccess::white; break;
            case BLACK | MAN : b.board[i] = EdAccess::black; break;
            case WHITE | KING: b.board[i] = EdAccess::white | EdAccess::king; break;
            case BLACK | KING: b.board[i] = EdAccess::black | EdAccess::king; break;
         }
      }
   }
   else
   {
      // при ходе черных "переворачиваем" доску
      for (i = 0; i < 32; i++)
      {
         switch (c[Map_32_to_45[31 - i]])
         {
            case FREE           : b.board[i] = EdAccess::empty; break;
            case WHITE | MAN : b.board[i] = EdAccess::black; break;
            case BLACK | MAN : b.board[i] = EdAccess::white; break;
            case WHITE | KING: b.board[i] = EdAccess::black | EdAccess::king; break;
            case BLACK | KING: b.board[i] = EdAccess::white | EdAccess::king; break;
         }
      }
   }

   return ED->GetResult(&b, 0);
}


//*************************************************
//*                                               *
//*               Kallisto support                *
//*                                               *
//*************************************************

int AnalyseMode = 0;
int PlayNow = 0;

int Board[46];
int Color;
int TimeRemaining;
int IncTime;

void Wait(int &v)
{
   while(v) Sleep(10);
}

void SquareToStr(unsigned short sq, char *s)
{
   static const int Square64[] = {
       0,  0,  0,  0,  0,
       7,  5,  3,  1,  0,
      14, 12, 10,  8,
      23, 21, 19, 17,  0,
      30, 28, 26, 24,
      39, 37, 35, 33,  0,
      46, 44, 42, 40,
      55, 53, 51, 49,  0,
      62, 60, 58, 56
   };

   sq = Square64[sq];
   s[0] = sq % 8 + 'a';
   s[1] = 8 - sq / 8 + '0';
   s[2] = 0;
}

void MoveToStr(move2 m, char *s)
{
   SquareToStr(m.m[0] & 63, s);
   for (int i = 2; i < m.l; i++) {
      strcat(s, ":");
      SquareToStr(m.m[i] & 63, s + strlen(s));
   }
   if (m.l > 2) strcat(s, ":");
   SquareToStr(m.m[1] & 63, s + strlen(s));
}

// Сделать ход move
// Формат ходов: "a3b4" и "a3:b4:d6:e7". Такой формат позволяет устранить все неоднозначности при взятиях
void KALLISTOAPI EI_MakeMove(char *move)
{
   if (AnalyseMode){
      PlayNow = 1;
      Wait(AnalyseMode);
   //EnterCriticalSection(&AnalyseSection);
   //LeaveCriticalSection(&AnalyseSection);
                                 }
   move2 ml[MAXMOVES];
   init_piece_lists(Board);
   int n = Gen_Captures(Board, ml, Color,1);
   if (!n) n = Gen_Moves(Board, ml, Color);
   for (int i = 0; i < n; i++) {
      char s[128];
      MoveToStr(ml[i], s);
      if (!strcmp(s, move)) {
         domove2(Board, &ml[i],Color);
         if (Color == WHITE) Color = BLACK;
         else Color = WHITE;
         return;
      }
   }
   MessageBox(0, "KestoG: move not found", move, MB_OK);
}

// Начать вычисления. После вернуть лучший ход
// Эта функция может выполняться как угодно долго
// Но надо иметь в виду количество оставшегося времени (см. EI_SetTimeControl и EI_SetTime)
char *KALLISTOAPI EI_Think()
{
   char dummy[256];
   PlayNow = 0;
   play = &PlayNow;
   int time_limit = TimeRemaining / 20 + IncTime;
   if   (time_limit * 2 > TimeRemaining) time_limit = TimeRemaining / 2;
   if ( time_limit  < 200 ){
   	 time_limit  = 200;
                                       }
   compute(Board, Color, time_limit, dummy);
   static char s[128];
   MoveToStr(bestrootmove, s);
   if (Color == WHITE) Color = BLACK;
   else Color = WHITE;
   return s;
}

// Здесь можно делать что угодно и как угодно долго
// Эта функция вызывается в момент когда противник думает над своим ходом
void KALLISTOAPI EI_Ponder()
{
   // здесь можно ничего и не делать :)
   return;
}

// Противник делает ход move
// Перед этим вызывалась функция Ponder
// Здесь сразу можно вернуть ход на основе вычиcлений сделанных в Ponder
// Можно подумать еще и только после этого вернуть ход
char *KALLISTOAPI EI_PonderHit(char *move)
{
   EI_MakeMove(move);
   return EI_Think();
}

// Инициализация движка
// si - см. выше описание PF_SearchInfo
// mem_lim - лимит памяти, которую может использовать движок
// здесь в основном имеется ввиду размер хэш-таблицы
void  KALLISTOAPI EI_Initialization(PF_SearchInfo si, int mem_lim)
 {
   pfSearchInfo = si;
   InitializeCriticalSection(&AnalyseSection);
   size = (unsigned int)mem_lim;
 }

// установка указателя на улучшенную функцию вывода инормации о переборе
void KALLISTOAPI EI_SetSearchInfoEx(PF_SearchInfoEx sie)
 {
	pfSearchInfoEx = sie;
 }

// Закончить вычисления и выйти из функций EI_Think, EI_Ponder, EI_PonderHit или EI_Analyse
void KALLISTOAPI EI_Stop()
{
   PlayNow = 1;
}

// Установить позицию pos на доске
// пример: начальная позиция bbbbbbbbbbbb........wwwwwwwwwwwww
// b - простая черная
// B - черная дамка
// w - простая белая
// W - белая дамка
// . - пустое поле
// поля перечисляются так: b8, d8, f8, h8, a7, c7, ..., a1, c1, e1, g1
// последний символ определяет очередность хода
// w - белые, b - черные
void KALLISTOAPI EI_SetupBoard(char *p)
{
   static const int Map[32] = {
       8,  7,  6,  5,
      13, 12, 11, 10,
      17, 16, 15, 14,
      22, 21, 20, 19,
      26, 25, 24, 23,
      31, 30, 29, 28,
      35, 34, 33, 32,
      40, 39, 38, 37
   };

   int i;
   for(i = 0; i < 46; i++) Board[i] = OCCUPIED;
   for (i = 0; i < 32; i++) {
      switch (p[i]) {
      case 'w': Board[Map[i]] = WHITE | MAN; break;
      case 'W': Board[Map[i]] = WHITE | KING; break;
      case 'b': Board[Map[i]] = BLACK | MAN; break;
      case 'B': Board[Map[i]] = BLACK | KING; break;
      case '.': Board[Map[i]] = FREE; break;
      }
   }
   if (p[32] == 'w') Color = WHITE;
   else Color = BLACK;
                  TTableInit(size);
                  EvalHashClear();
                  searches_performed_in_game = 0;
                  Create_HashFunction();
                  ClearHistory();
                            }

void KALLISTOAPI EI_NewGame()
{
          EI_SetupBoard("bbbbbbbbbbbb........wwwwwwwwwwwww");
//                  TTableInit(size);
//                  searches_performed_in_game = 0;
//                  Create_HashFunction();
}

// Установить контроль времени
// time минут на партию
// inc секунд - бонус за каждый сделанный ход (часы Фишера)
void KALLISTOAPI EI_SetTimeControl(int time, int inc)
{
   TimeRemaining = time * 60 * 1000;
   IncTime = inc;
}

// Установить время в миллисекундах оставшееся на часах
// time - свое время
// otime - время противника
void KALLISTOAPI EI_SetTime(int time, int otime)
{
   TimeRemaining = time;
}

// Вернуть название движка
char *KALLISTOAPI EI_GetName()
{
   return "KestoG 1.5 I";
}

// Вызывается перед выгрузкой движка
void KALLISTOAPI EI_OnExit()
{
    if ( ttable != NULL ){
    free(ttable);
    ttable = NULL;
                                     }
}

// Анализировать текущую позицию
// Выход из режима анализа осуществляется при получении команд Stop или МакеMove
void KALLISTOAPI EI_Analyse()
{
   AnalyseMode = true;
   char dummy[256];
   PlayNow = 0;
   play = &PlayNow;
   EnterCriticalSection(&AnalyseSection);
          TTableInit(size);
          EvalHashClear();
          searches_performed_in_game = 0;
          Create_HashFunction();
          ClearHistory();
   compute(Board, Color, 2000000000, dummy); // infinity thinking
   undomove(Board, &bestrootmove,Color);
   AnalyseMode = false;
   LeaveCriticalSection(&AnalyseSection);
}


// функция интерфейса экспортируемая из dll
void KALLISTOAPI EI_EGDB(EdAccess *eda)
{
   ED = eda;
   if (ED)
   {
        EdPieces = ED->Load("russian");
        if (strstr(ED->GetBaseType(), "nocaptures")) EdNocaptures = true;
        //MessageBox(0, "Initialization-nocaptures", "", MB_OK);
   }
   //EdPieces = 0;  // comment this line out if EGDB usage works
   //MessageBox(0, "Initialization", "", MB_OK);
 }