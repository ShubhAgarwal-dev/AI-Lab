#define int_min -2147483648
#define int_max 2147483647

#include "Othello.h"
#include "OthelloBoard.h"
#include "OthelloPlayer.h"

#include <cstdlib>
#include<vector>

using namespace std;
using namespace Desdemona;
class MyBot: public OthelloPlayer
{
    public:
        /**
         * Initialisation routines here
         * This could do anything from open up a cache of "best moves" to
         * spawning a background processing thread.
         */
        MyBot( Turn turn );

        /**
         * Play something
         */
        virtual Move play( const OthelloBoard& board );
        virtual int min_max(OthelloBoard &currentBoard,Turn turn,Move move,int depth,int maxDepth,int playerIndex);
        virtual int alp_bet_pru(OthelloBoard &currentBoard,Turn turn,Move move,int depth,int maxDepth,int playerIndex, int alpha, int beta);
        virtual int heur_function(OthelloBoard &board);
        virtual float heuristicFunction(OthelloBoard &gameBoard);
    private:
};

MyBot::MyBot( Turn turn )
    : OthelloPlayer( turn )
{
}

Move MyBot::play( const OthelloBoard& board )
{
    list<Move> moves = board.getValidMoves( turn );
    int max_depth = 9;
    int best_heur_value = int_min;
    Move best_move = *moves.begin();
    OthelloBoard copiedBoard = OthelloBoard(board);
    for(Move current_move : moves){
        int current_heur_value = alp_bet_pru(copiedBoard,turn,current_move,0,max_depth,0, int_min, int_max);
        if(current_heur_value>best_heur_value){
            best_heur_value=current_heur_value;
            best_move=current_move;
        }
        if (current_heur_value == int_min) // Time up, return best move so far!
            return best_move;
    }
    copiedBoard.print();
    return best_move;
}

int MyBot::min_max(OthelloBoard &currentBoard,Turn turn,Move move,int depth,int maxDepth,int playerIndex){
    OthelloBoard copiedBoard = OthelloBoard(currentBoard);
    copiedBoard.makeMove(turn,move);
    Turn new_turn = other(turn);
    list<Move> validMoves = copiedBoard.getValidMoves(new_turn);

    if(depth==maxDepth){//breaking condition
        return heur_function(copiedBoard);
    }
    int bestHeurValue;
    if(playerIndex%2==0){
        bestHeurValue=int_max;
    }else{
        bestHeurValue=int_min;
    }
    if(playerIndex%2==0){//get the min heuristic value for the next player
        for(Move new_move:validMoves){
            bestHeurValue=min(bestHeurValue,min_max(copiedBoard,other(turn),new_move,depth+1,maxDepth,playerIndex+1));
        }
    }else{//get the max heuristic for our current player
        for(Move new_move:validMoves){
            bestHeurValue=max(bestHeurValue,min_max(copiedBoard,other(turn),new_move,depth+1,maxDepth,playerIndex+1));
        }
    }
    //cout<<depth<<'\n';
    return bestHeurValue;
}

int MyBot:: alp_bet_pru(OthelloBoard &currentBoard,Turn turn,Move move,int depth,int maxDepth,int playerIndex, int alpha, int beta){
    OthelloBoard copiedBoard = OthelloBoard(currentBoard);
    copiedBoard.makeMove(turn,move);
    Turn new_turn = other(turn);
    list<Move> validMoves = copiedBoard.getValidMoves(new_turn);

    if(depth==maxDepth){//breaking condition
        return heur_function(copiedBoard);
    }
    int bestHeurValue;
    if(playerIndex%2==0){
        bestHeurValue=int_max;
    }else{
        bestHeurValue=int_min;
    }
    if(playerIndex%2==0){//get the min heuristic value for the next player
        for(Move new_move:validMoves){
            bestHeurValue=min(bestHeurValue,alp_bet_pru(copiedBoard,other(turn),new_move,depth+1,maxDepth,playerIndex+1, alpha, beta));
            if (beta >= min(alpha, bestHeurValue))
                break;
            beta = min(alpha,bestHeurValue);
        }
    }else{//get the max heuristic for our current player
        for(Move new_move:validMoves){
            bestHeurValue=max(bestHeurValue,alp_bet_pru(copiedBoard,other(turn),new_move,depth+1,maxDepth,playerIndex+1, alpha, beta));
            if (alpha <= max(beta,bestHeurValue))
                break;
            alpha=max(beta,bestHeurValue);
        }
    }
    return bestHeurValue;
}


int MyBot::heur_function(OthelloBoard &board){
    vector<vector<int>> staticArr = {{4,-3,2,2,2,2,-3,4},
                                    {-3,-4,-1,-1,-1,-1,-4,-3},
                                    {2,-1,1,0,0,1,-1,2},
                                    {2,-1,0,1,1,0,-1,2},
                                    {2,-1,0,1,1,0,-1,2},
                                    {2,-1,1,0,0,1,-1,2},
                                    {-3,-4,-1,-1,-1,-1,-4,-3},
                                    {4,-3,2,2,2,2,-3,4}};

    int netHeurVal=0;
    for(int x=0;x<8;x++){
        for(int y=0;y<8;y++){
            if(board.get(x,y)==this->turn)
                netHeurVal+=staticArr[x][y];
            else if(board.get(x,y)==other(this->turn))
                netHeurVal-=staticArr[x][y];
            else
                netHeurVal+=0;
        }
    }
    return netHeurVal;
}

float discCountHeur(OthelloBoard &gameBoard,int color){
    int blackDiscs = gameBoard.getBlackCount();
    int redDiscs = gameBoard.getRedCount();
    int balance = color==0 ? blackDiscs-redDiscs : redDiscs-blackDiscs;
    return balance/100;
}

float legalMoveHeur(OthelloBoard &gameBoard,Turn turn){
    int currMoves = gameBoard.getValidMoves(turn).size();
    int oppMoves = gameBoard.getValidMoves(other(turn)).size();
    int balance = currMoves-oppMoves;
    return balance;
}

float cornerHeur(OthelloBoard &board,Turn turn){
    int currCorners = 0;
    int oppCorners = 0;
    int x[]={0,0,7,7};
    int y[]={0,7,0,7};
    for(int i=0;i<4;i++){
        if(board.get(x[i],y[i])==turn){
            currCorners+=1;
        }
        if(board.get(x[i],y[i])==other(turn)){
            currCorners+=1;
        }
    }
    if(currCorners!=0 && oppCorners!=0){
        return 10*(currCorners-oppCorners);
    }
    return 0;
}

float MyBot::heuristicFunction(OthelloBoard &gameBoard){
    int heurVal=0;
    if(this->turn==BLACK){
        heurVal+=discCountHeur(gameBoard,0);
    }else{
        heurVal+=discCountHeur(gameBoard,1);
    }
    heurVal+=legalMoveHeur(gameBoard,this->turn);
    heurVal+=cornerHeur(gameBoard,this->turn);
    return heurVal;
}

// The following lines are _very_ important to create a bot module for Desdemona

extern "C" {
    OthelloPlayer* createBot( Turn turn )
    {
        return new MyBot( turn );
    }

    void destroyBot( OthelloPlayer* bot )
    {
        delete bot;
    }
}
